import logging
import os
from io import BytesIO

import requests
from PIL import Image

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def decode_request(request) -> Image.Image | None:
    """
    Decodes an incoming request to extract an image.

    Args:
        request: The incoming request object. Expected to have a "content" key
                 which can be a URL string or a file-like object from an upload.

    Returns:
        An PIL.Image.Image object if successful, None otherwise.
    """
    file_content = request.get("content")

    if file_content is None:
        logger.warning("Request content is missing.")
        return None

    if isinstance(file_content, str) and "http" in file_content:
        # Handle URL input
        url = file_content.replace("localhost:3210", "backend:3210")  # HACK
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes
            image = Image.open(BytesIO(response.content))
            logger.debug("Successfully processed URL input.")
            return image
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch image from URL {url}: {e}")
            return None
        except IOError as e:
            logger.error(f"Failed to open image from URL {url} content: {e}")
            return None
    elif hasattr(file_content, "file") and not isinstance(file_content, str):
        try:
            # We've narrowed the type, so file_content.file is now more safely accessed
            # The linter knows if we reach here, it's not a str.
            file_bytes = file_content.file.read()
            image = Image.open(BytesIO(file_bytes))
            logger.debug("Successfully processed file input.")
            return image
        except (
            AttributeError
        ):  # Keep this for robustness in case file_content.file isn't callable
            logger.error(
                "Failed to access 'file' attribute or 'read' method from file_content."
            )
            return None
        except IOError as e:
            logger.error(f"Failed to open image from file content: {e}")
            return None
        finally:
            # Ensure the file handle is closed if it's an upload object
            if hasattr(file_content.file, "close"):
                file_content.file.close()
    else:
        # Catch-all for unexpected types
        logger.warning(
            f"Unexpected content type or format for request: {type(file_content)}. "
            "Expected a URL string or a file upload object."
        )
        return None
