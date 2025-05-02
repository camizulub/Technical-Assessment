from setuptools import setup, find_packages

setup(
    name="trading_package",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "plotly>=5.3.0",
        "pydantic>=2.0.0",
    ],
    author="Camilo Zuluaga",
    author_email="czuluaga@sunvalleyinv.com",
    description="A modular package for trading strategy development and backtesting",
    keywords="trading, finance, backtesting, stocks, crypto, analysis",
    python_requires=">=3.8",
) 