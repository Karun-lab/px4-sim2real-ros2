from setuptools import find_packages, setup

package_name = 'sim2real'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='karun',
    maintainer_email='karunashok16@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "offboard_ctrl=sim2real.offboard_ctrl:main",
            "mav_offboard_ctrl=sim2real.mav_offboard_ctrl:main",
            "test_rate_cmd=sim2real.test_rate_cmd:main",
            "keyboard_teleop=sim2real.keyboard_teleop:main",
            "gstreamer=sim2real.gstreamer:main",
            "rl_inference_node=sim2real.rl_inference_node:main",
            "tello_node = sim2real.tello_node:main",
            "flight_test_offboard=sim2real.flight_test_offboard:main",
            "ICM_Inference=sim2real.ICM_Inference:main",
            "tello_icm_bridge_node=sim2real.tello_icm_bridge_node:main",
            "px4_icm_node=sim2real.px4_icm_node:main",
        ],
    },
)
