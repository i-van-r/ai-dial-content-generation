import asyncio
from datetime import datetime

from task._models.custom_content import Attachment
from task._utils.constants import API_KEY, DIAL_URL, DIAL_CHAT_COMPLETIONS_ENDPOINT
from task._utils.bucket_client import DialBucketClient
from task._utils.model_client import DialModelClient
from task._models.message import Message
from task._models.role import Role


class Size:
    """
    The size of the generated image.
    """
    square: str = '1024x1024'
    height_rectangle: str = '1024x1792'
    width_rectangle: str = '1792x1024'


class Style:
    """
    The style of the generated image. Must be one of vivid or natural.
     - Vivid causes the model to lean towards generating hyper-real and dramatic images.
     - Natural causes the model to produce more natural, less hyper-real looking images.
    """
    natural: str = "natural"
    vivid: str = "vivid"


class Quality:
    """
    The quality of the image that will be generated.
     - ‘hd’ creates images with finer details and greater consistency across the image.
    """
    standard: str = "standard"
    hd: str = "hd"


async def _save_images(attachments: list[Attachment]):
    async with DialBucketClient(api_key=API_KEY, base_url=DIAL_URL) as bucket_client:
        for idx, attachment in enumerate(attachments):
            if attachment.url:
                print(f"\nDownloading image {idx + 1}/{len(attachments)}...")
                image_bytes = await bucket_client.get_file(attachment.url)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_name = f"generated_image_{timestamp}_{idx + 1}.png"

                with open(file_name, 'wb') as f:
                    f.write(image_bytes)

                print(f"Image saved locally as: {file_name}")


def start() -> None:
    print("\n" + "=" * 80)
    print("Text-to-Image Generation Task")
    print("=" * 80 + "\n")

    deployment_name = "dall-e-3"
    print(f"Using model: {deployment_name}")
    print("Note: If this model doesn't work, try these alternatives:")
    print("  - dalle-3")
    print("  - dall-e")
    print("  - imagegeneration")
    print("Or check available models at: https://ai-proxy.lab.epam.com/openai/models\n")

    client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name=deployment_name,
        api_key=API_KEY
    )

    prompt = "Sunny day on Bali"
    print(f"Generating image for prompt: '{prompt}'\n")

    custom_fields = {
        "size": Size.square,
        "quality": Quality.hd,
        "style": Style.vivid
    }

    message = Message(
        role=Role.USER,
        content=prompt
    )

    print("Sending request to image generation model...\n")
    response = client.get_completion(
        messages=[message],
        custom_fields=custom_fields
    )

    print("\n" + "=" * 80)
    print("Image generation completed!")
    print("=" * 80)
    print(f"\nResponse content: {response.content}\n")

    if response.custom_content and response.custom_content.attachments:
        print(f"Number of generated images: {len(response.custom_content.attachments)}\n")

        for idx, attachment in enumerate(response.custom_content.attachments):
            print(f"Image {idx + 1}:")
            print(f"  - Title: {attachment.title}")
            print(f"  - URL: {attachment.url}")
            print(f"  - Type: {attachment.type}")

        print("\n" + "=" * 80)
        print("Saving images locally...")
        print("=" * 80)

        asyncio.run(_save_images(response.custom_content.attachments))

        print("\n" + "=" * 80)
        print("All images saved successfully!")
        print("=" * 80)
        print("\nConfiguration used:")
        print(f"  - Size: {custom_fields['size']}")
        print(f"  - Quality: {custom_fields['quality']}")
        print(f"  - Style: {custom_fields['style']}")
        print("\nTip: You can modify custom_fields to change image parameters!")
        print("Try different sizes: square, height_rectangle, width_rectangle")
        print("Try different styles: natural, vivid")
        print("Try different quality: standard, hd")
        print("=" * 80 + "\n")
    else:
        print("\nWarning: No images were generated in the response.\n")


start()
