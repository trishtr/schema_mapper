from setuptools import setup, find_packages

setup(
    name="schema_mapper",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.0",
        "sentence-transformers>=2.2.0",
        "chromadb>=0.4.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "torch>=1.9.0",
        "python-dotenv>=0.19.0",
        "aioredis>=2.0.0",
        "cachetools>=5.0.0",
        "SQLAlchemy>=1.4.0",
        "aiosqlite>=0.17.0",
        "python-jose>=3.3.0",
        "passlib>=1.7.4",
        "bcrypt>=3.2.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.18.0",
            "httpx>=0.23.0"
        ]
    },
    python_requires=">=3.8",
    author="Trisha",
    author_email="trisha@example.com",
    description="An intelligent schema mapping system using embedding models and vector search",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/trishtr/schema_mapper",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10"
    ]
) 