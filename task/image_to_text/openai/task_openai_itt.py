import base64
from pathlib import Path

from task._utils.constants import API_KEY, DIAL_CHAT_COMPLETIONS_ENDPOINT
from task._utils.model_client import DialModelClient
from task._models.role import Role
from task.image_to_text.openai.message import ContentedMessage, TxtContent, ImgContent, ImgUrl


def start() -> None:
    project_root = Path(__file__).parent.parent.parent.parent
    image_path = project_root / "dialx-banner.png"

    # 1. Create DialModelClient
    client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name="gpt-4o",
        api_key=API_KEY
    )

    # 2. Call client to analyze image with base64 encoded format (Method 1)
    print("\n" + "=" * 80)
    print("Analyzing image with BASE64 encoded format (dialx-banner.png)")
    print("=" * 80 + "\n")

    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    base64_message = ContentedMessage(
        role=Role.USER,
        content=[
            TxtContent(text="What's in this image? Please describe it in detail."),
            ImgContent(image_url=ImgUrl(url=f"data:image/png;base64,{base64_image}"))
        ]
    )

    response_base64 = client.get_completion(messages=[base64_message])
    print(f"\nResponse: {response_base64.content}\n")

    # 3. Call client to analyze image with data URI format (Method 2)
    # Note: External URLs may not work with DIAL proxy due to network restrictions.
    # Using base64 data URI is the recommended approach for DIAL Core.
    print("\n" + "=" * 80)
    print("Analyzing the same image with data URI format (demonstrating URL syntax)")
    print("=" * 80 + "\n")

    # Reuse the same image in data URI format to demonstrate URL parameter
    data_uri_message = ContentedMessage(
        role=Role.USER,
        content=[
            TxtContent(text="Describe the color scheme and branding elements in this image."),
            ImgContent(image_url=ImgUrl(url=f"data:image/png;base64,{base64_image}"))
        ]
    )

    response_data_uri = client.get_completion(messages=[data_uri_message])
    print(f"\nResponse: {response_data_uri.content}\n")

    print("\n" + "=" * 80)
    print("✅ Both methods completed successfully!")
    print("Note: External URLs are not supported by DIAL proxy due to security restrictions.")
    print("Use base64 data URIs for reliable image analysis with DIAL Core.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    start()
