from setuptools import setup, find_packages

setup(
    name="ggcnn",
    version="0.1",
    description="GGCNN grasp generation",
    license="MIT",
    author="Autonomous Learning Robots @ KIT",
    url="https://alr.anthropomatik.kit.edu/",
    packages=find_packages(),
    intstall_requires=[
        "numpy",
        "opencv-python",
        "matplotlib",
        "scikit-image",
        "imageio",
        "torch",
        "torchvision",
        "torchsummary",
        "tensorboardX",
    ],
    extras_require={
        "dev": [
            "black",
        ],
    },
)
