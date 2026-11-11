"""The images feature: what the agent is told about images, and the image window."""
from deep_agents_demo.core.feature import Feature
from deep_agents_demo.features.images.window import ImageWindow


def images_feature(prefix: str = "") -> Feature:
    prompt = ("The user can attach PNG or JPG images: they arrive with their message and are saved "
              f"under {prefix}/uploads/, where read_file shows them again. read_file also shows you "
              "any image file. When you create an image or chart, open it with read_file and check "
              "it before saying it's done: labels readable, nothing cut off or overlapping, not "
              "empty, showing what was asked. If it's wrong, fix the code and check again. ")
    return Feature(name="images", prompt=prompt, middleware=(ImageWindow(),))
