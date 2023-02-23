from setuptools import setup, find_packages

setup(
    name="ggcnn",
    version="0.1",
    description="GGCNN grasp generation",
    license="MIT",
    author="Autonomous Learning Robots @ KIT",
    url="https://alr.anthropomatik.kit.edu/",
    package_data={"models": ["*"]},
    packages=find_packages(),
    install_requires=[
        "numpy<1.24",  # ros_numpy does not support numpy 1.24 yet: https://github.com/eric-wieser/ros_numpy/pull/32
        "pyyaml",
        "trimesh",
    ],
)