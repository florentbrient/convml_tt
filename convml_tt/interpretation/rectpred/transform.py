"""
Utilities for applying transforms on embedding dimensions for data on
reactangular domains
"""
try:
    import hdbscan

    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
import sklearn.cluster
import xarray as xr


def _apply_transform_function(da, fn, transform_name):
    # stack all other dims apart from the `emb_dim`
    dims = list(da.dims)
    if "emb_dim" in dims:
        dims.remove("emb_dim")
    elif "pca_dim" in dims:
        dims.remove("pca_dim")
    else:
        raise NotImplementedError(da.dims)
    da_stacked = da.stack(dict(n=dims))
    arr = fn(X=da_stacked.T)
    if len(arr.shape) == 2:
        dims = ("n", "{}_dim".format(transform_name))
    else:
        dims = ("n",)
    return xr.DataArray(arr, dims=dims, coords=dict(n=da_stacked.n)).unstack("n")


# flake8: noqa: C901
def apply_transform(
    da, transform_type, pretrained_model=None, return_model=False, **kwargs
):
    """
    Apply transform to xr.DataArray `da` representing embedding-vectors (or
    other 1D vector data) on a 2D rectangular domain (with indecies given by
    `i0` `j0`). The type of transform to applied should be provided by
    `transform_type` and be one of `pca`, `kmeans`, `hdbscan`.

    For `transform_type=="pca"` you can provide a pretrained transform model
    with `pretrained_model`
    """
    add_meta = None
    model = None

    if "hdbscan" in transform_type and not HAS_HDBSCAN:
        raise Exception("To use `hdbscan` you will need to install it with pip first")

    if transform_type == "kmeans":
        fn_transform = sklearn.cluster.KMeans(**kwargs).fit_predict
    elif transform_type == "pca":
        if pretrained_model is not None:
            model = pretrained_model
            fn_transform = model.transform
        else:
            model = sklearn.decomposition.PCA(**kwargs)
            fn_transform = model.fit_transform

        def add_meta(da):
            da["explained_variance"] = (
                "{}_dim".format(transform_type),
                model.explained_variance_ratio_,
            )

    elif transform_type == "hdbscan":
        if not HAS_HDBSCAN:
            raise Exception(
                "To use `hdbscan` you will need to install it with pip first"
            )
        model = hdbscan.HDBSCAN(core_dist_n_jobs=-1, **kwargs)

        def fn_transform(X):
            return model.fit(X).labels_

        def add_meta(da):
            return xr.DataArray(
                model.probabilities_,
                dims=("n"),
                coords=dict(n=da.stack(dict(n=da.dims)).n),
            ).unstack("n")

    elif transform_type == "isomap":
        if pretrained_model is not None:
            model = pretrained_model
            fn_transform = model.transform
        else:
            model = sklearn.manifold.Isomap(**kwargs)
            fn_transform = model.fit_transform

    elif transform_type == "pca_hdbscan":
        try:
            pca__n_components = kwargs.pop("pca__n_components")
        except KeyError:
            raise Exception(
                "To use HDBSCAN with PCA analysis first you need"
                " provide the number of PCA components to keep with"
                " the `pca__n_components` argument"
            )
        pca_model = sklearn.decomposition.PCA(n_components=pca__n_components)
        hdbscan_model = hdbscan.HDBSCAN(core_dist_n_jobs=-1, **kwargs)
        model = hdbscan_model

        def fn_transform(X):
            X1 = pca_model.fit_transform(X)
            return hdbscan_model.fit(X1).labels_

        def add_meta(da):
            da.attrs["notes"] = "used PCA before HDBSCAN"

    else:
        raise NotImplementedError(transform_type)

    da_transformed = _apply_transform_function(
        da=da, fn=fn_transform, transform_name=transform_type
    )

    da_transformed.attrs.update(da.attrs)

    if add_meta is not None:
        add_meta(da_transformed)

    da_transformed.attrs.update(da.attrs)
    da_transformed.name = "emb"
    for v in ["i0", "j0"]:
        if v in da:
            da_transformed[v] = da[v]
    da_transformed.attrs["transform_type"] = transform_type
    if kwargs:
        s = ",".join([f"{k}={v}" for (k, v) in kwargs])
        da_transformed.attrs["transform_extra_args"] = s

    # explicitly copy over coords and attrs that we want to retain
    for c in ["image_path", "src_data_path", "tile_id"]:
        if c in da.coords or c in da.attrs:
            da_transformed[c] = da[c]

    if return_model:
        return da_transformed, model
    else:
        return da_transformed
