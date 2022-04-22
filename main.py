import os
import argparse
import sys
sys.path.insert(0, 'external/python-seg2mesh')
sys.path.insert(1, 'external/python-spline-fitting-toolbox')
import tifffile
from scipy import ndimage
from scipy.ndimage.morphology import distance_transform_edt
from ma_estimations.utils import *
from ma_estimations.graph import create_graph, sample_points
from ma_estimations.spline import cut_spline, interpolate


def main():
    files = os.listdir(args.vol_dir)
    files = [f for f in files if '.tif' in f]

    endpoints = load_points(args.endpoints_file)
    midpoints = load_points(args.midpoints_file)

    desc_dict = {}
    for num_file, f in enumerate(files):
        print(f'[{num_file}] processing file {f}')
        key = f[:-4]
        pts = get_points(
            endpoints[key + str(1)], endpoints[key + str(2)]
        )
        res = get_resolution(endpoints[key + str(1)])
        M = int(endpoints[key + str(1)]['M'])

        vol = tifffile.imread(
            os.path.join(args.vol_dir, f)
        ).astype('float32')
        vol = ndimage.zoom(vol, args.zoom, order=0)
        vol[vol > 0] = 1
        mesh = voxel_to_mesh(vol)
        vol = mesh_to_voxel(mesh, vol.shape)
        dst = distance_transform_edt(vol)
        max_dst = dst.max()
        digraph = create_graph(dst)
        pts = map_points(digraph, pts[0], pts[1], args.zoom)
        sampled_points = sample_points(digraph, pts[0], pts[1])
        contour, reflection_points, spline = interpolate(
            sampled_points, M, args.sampling_rate
        )
        contour = cut_spline(contour, pts[0], pts[1])
        try:
            landmark_pts = get_points(
                midpoints[key + str(1)], midpoints[key + str(2)]
            )
            landmark_pts = map_points(digraph, landmark_pts[0], landmark_pts[1], args.zoom)
            print('Computing diameter and radii...')
            radii_diameter = compute_radius_and_diameter(
                landmark_pts[0],
                landmark_pts[1],
                sampled_points,
                args.sampling_rate,
                res,
                args.zoom,
                min_plane_sz=max_dst,
                mesh=mesh,
                M=M
            )
        except (TypeError, ValueError) as e:
            print('Diameter and radii computation failed!')
            print(e)
            radii_diameter = (-1.0, -1.0, -1.0, -1.0)

        del digraph

        print('Diameter: ', radii_diameter[3])

        print('Computing length...')
        length = compute_length(
            pts[0], pts[1], sampled_points, args.sampling_rate, res, args.zoom, M
        )
        print('Averaged length:', length)

        descriptors = {
            key: {'length': float(length),
                  'min_radius': float(radii_diameter[0]),
                  'max_radius': float(radii_diameter[1]),
                  'mean_radius': float(radii_diameter[2]),
                  'diameter': float(radii_diameter[3])}
        }
        desc_dict.update(descriptors)

        save_descriptors(args.save_to, desc_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--vol_dir",
                        type=str,
                        required=True,
                        help="Path to directory with image volumes.")
    parser.add_argument("--endpoints_file",
                        type=str,
                        required=True,
                        help="Path to file with endpoints.")
    parser.add_argument("--midpoints_file",
                        type=str,
                        required=True,
                        help="Path to file with midpoints.")
    parser.add_argument('--zoom',
                        type=float,
                        default=.4,
                        help="Downsampling factor (<1.0, default: 0.15)")
    parser.add_argument('--sampling_rate',
                        type=int,
                        default=200,
                        help="Sampling rate of the spline.")
    parser.add_argument('--save_to',
                        type=str,
                        help="Path + name of output file (e.g. 'data/InVivo/descriptors.csv)")
    args = parser.parse_args()
    main()
