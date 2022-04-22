import csv
import numpy as np
from scipy.spatial.distance import cdist
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from joblib import Parallel, delayed
from skimage import measure
from vedo import Plane, Mesh
from seg2mesh import _getLargestCC, ndarray2vtkMesh, decimation, smooth
from .spline import interpolate


def voxel_to_mesh(segmentation, _step_size=2, _reduction=.1, _smoothing=True):
    assert np.unique(segmentation).tolist() == [0, 1]

    obj = segmentation == 1
    obj = obj.astype(float)
    obj = _getLargestCC(obj)
    vertices, faces, normals, _ = measure.marching_cubes(
        obj, 0, step_size=_step_size
    )

    vtk_poly = ndarray2vtkMesh(vertices, faces.astype(int))
    assert vtk_poly is not None

    if _reduction > 0:
        vtk_poly = decimation(vtk_poly, reduction=_reduction)

    if _smoothing:
        vtk_poly = smooth(vtk_poly, edgesmoothing=False)

    return vtk_poly


def mesh_to_voxel(mesh, dim, spacing=(1, 1, 1)):
    pd = mesh
    whiteImage = vtk.vtkImageData()
    whiteImage.SetSpacing(spacing)
    whiteImage.SetDimensions(dim)
    whiteImage.SetExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1)

    origin = [0, 0, 0]
    whiteImage.SetOrigin(origin)
    whiteImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    count = whiteImage.GetNumberOfPoints()
    for i in range(count):
        whiteImage.GetPointData().GetScalars().SetTuple1(i, 255)

    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(pd)
    pol2stenc.SetOutputOrigin(origin)
    pol2stenc.SetOutputSpacing(spacing)
    pol2stenc.SetOutputWholeExtent(whiteImage.GetExtent())
    pol2stenc.Update()

    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(whiteImage)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(0)
    imgstenc.Update()

    vtk_image_data = imgstenc.GetOutput()
    narray_shape = tuple(reversed(vtk_image_data.GetDimensions()))
    img_narray = vtk_to_numpy(
        vtk_image_data.GetPointData().GetScalars()
    ).reshape(narray_shape)
    img_narray = np.transpose(img_narray, axes=[2, 1, 0])
    return img_narray


def load_points(path_to_points):
    points = {}
    with open(path_to_points, mode='r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            filename = row['File_name'] + str(row['Point'])
            del row['File_name'], row['Point']
            points.update({filename: row})
    return points


def get_points(pt1, pt2):
    pt1 = (float(pt1['Position Z']),
           float(pt1['Position Y']),
           float(pt1['Position X']))
    pt2 = (float(pt2['Position Z']),
           float(pt2['Position Y']),
           float(pt2['Position X']))
    return (np.array(pt1), np.array(pt2))


def map_points(digraph, pt1, pt2, zoom):
    pts = []
    for pt in [pt1, pt2]:
        u = tuple([int(p * zoom) for p in pt])
        u = min(
            digraph.nodes,
            key=lambda t: (t[0] - u[0])**2 + (t[1] - u[1])**2 + (t[2] - u[2])**2)
        pts += [u]
    return pts


def get_resolution(res_dict):
    if 'px_size_X' in res_dict:
        res = float(res_dict['px_size_X'])
    elif 'X_px_size_um' in res_dict:
        res = float(res_dict['X_px_size_um'])
    return res


def compute_length(pt1, pt2, sampled_points, sampling_rate, res, zoom, M):
    c, rp, spline = interpolate(sampled_points, M, sampling_rate)
    length, _ = spline.arc_length(pt1, pt2, sampling_rate)
    length = length * (1 / zoom) * res
    if length > 1000:
        length = -1.0
    return length


def intersect(t, spline, sampling_rate, min_plane_sz, pt1, pt2):
    plane = Plane(
        pos=spline.parameter_to_world(t / sampling_rate),
        normal=spline.parameter_to_world(t / sampling_rate, dt=True),
        sx=min_plane_sz + 200
    )
    dst = np.linalg.norm(plane.closestPoint(pt1) - pt1) + \
          np.linalg.norm(plane.closestPoint(pt2) - pt2)
    return dst, t


def compute_radius_and_diameter(
    pt1,
    pt2,
    sampled_points,
    sampling_rate,
    res,
    zoom,
    min_plane_sz,
    mesh,
    M
):
    c, rp, spline = interpolate(sampled_points, M, sampling_rate)
    N = (sampling_rate * (M - 1)) + 1
    contour = np.array(
        [spline.parameter_to_world(float(i) / float(sampling_rate)) for i in range(0, N)]
    )
    src_idx = np.linalg.norm(contour - pt1, axis=1).argmin()
    trg_idx = np.linalg.norm(contour - pt2, axis=1).argmin()
    idx1 = src_idx if src_idx < trg_idx else trg_idx
    idx2 = src_idx if src_idx >= trg_idx else trg_idx

    dst_idx_list = Parallel(n_jobs=4, backend='threading')(
        delayed(intersect)(t, spline, sampling_rate, min_plane_sz, pt1, pt2)
        for t in range(idx1, idx2, 2)
    )

    distances = np.asarray([e[0] for e in dst_idx_list])
    indices = np.asarray([e[1] for e in dst_idx_list])
    idx = indices[distances.argmin()]
    try:
        midpoint = spline.parameter_to_world(idx / float(sampling_rate))
        plane = Plane(
            pos=midpoint,
            normal=spline.parameter_to_world(idx / float(sampling_rate), dt=True),
            sx=min_plane_sz + 200
        )
        points = Mesh(mesh).intersectWith(plane).points()
        diameter = cdist(points, points).max() * res
        dst = np.linalg.norm(points - midpoint, axis=1)
        rmin = dst.min() * res
        rmax = dst.max() * res
        rmean = dst.mean() * res
    except TypeError as e:
        print(e)
        rmin, rmax, rmean = (-1, -1, -1)
        diameter = -1

    return rmin, rmax, rmean, diameter


def save_descriptors(path_to_save, desc_dict):
    with open(path_to_save, mode='w') as csvfile:
        fields = [
            'filename',
            'length',
            'min_radius',
            'max_radius',
            'mean_radius',
            'diameter'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for k, v in desc_dict.items():
            writer.writerow(
                {'filename': k,
                 'length': v['length'],
                 'min_radius': v['min_radius'],
                 'max_radius': v['max_radius'],
                 'mean_radius': v['mean_radius'],
                 'diameter': v['diameter']}
            )
