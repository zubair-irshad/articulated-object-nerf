import sapien
from data_utils import *
import argparse

def main(args):
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer(offscreen_only=True)
    engine.set_renderer(renderer)

    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    
    urdf_path = args.urdf_file
    asset = loader.load_kinematic(str(urdf_path))
    # asset = loader.load(urdf_path)
    assert asset, 'URDF not loaded.'


    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

    near, far = 0.1, 100
    width, height = args.resolution
    # width, height = 640, 480
    camera = scene.add_camera(
        name="camera",
        width=width,
        height=height,
        fovy=np.deg2rad(35),
        near=near,
        far=far,
    )

    output_path = args.output_path
    P(output_path).mkdir(exist_ok=True)


    splits = ('train', 'test', 'val')

    gen_articulated_object_nerf_s1(100, 4, 'train', camera, asset, scene, object_path=output_path)
    gen_articulated_object_nerf_s1(50, 4, 'test', camera, asset, scene, object_path=output_path)
    gen_articulated_object_nerf_s1(50, 4, 'val', camera, asset, scene, object_path=output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your program.")
    
    # Define command-line arguments
    parser.add_argument("--urdf_file", type=str, required=True, help="Path to urdf file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory.")
    parser.add_argument("--resolution", type=list, default=[640, 480] , help="list of resolution, [w, h]")
    
    args = parser.parse_args()
    main(args)