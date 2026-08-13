from setuptools import setup
import os
from glob import glob

package_name = 'costmap'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config',
            glob('config/*.yaml')),
        ('share/' + package_name + '/launch',
            glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Griphonyx Team',
    maintainer_email='griphonyx@example.com',
    description='Local costmap node for Griphonyx UAV obstacle avoidance',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'costmap_node = costmap.costmap_node:main',
        ],
    },
)
