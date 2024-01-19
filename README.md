## Install Python Dependencies
```
pip3 install -r requirements.txt
```
## If ModuleNotFoundError: No module named 'image_capnp'
git submodule update --init --recursive

## Design Principle
1. The viewer connects to the rest of the system through eCAL middleware, hence it can run on the host, while the rest of the system runs on embedded headless system
2. The viewer can itself runs in headless mode as well. Although currently we will be implementing it in Python, so not runtime efficient yet
