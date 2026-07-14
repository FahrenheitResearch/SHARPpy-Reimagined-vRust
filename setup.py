"""Wheel tagging for the bundled native Rusty Weather executables."""

from setuptools import Distribution, setup


class BinaryDistribution(Distribution):
    """Mark wheels as platform-specific even though binaries are package data."""

    def has_ext_modules(self):
        return True


setup(distclass=BinaryDistribution)
