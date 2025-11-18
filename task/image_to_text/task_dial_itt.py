import asyncio
from io import BytesIO
from pathlib import Path

from task._models.custom_content import Attachment, CustomContent
from task._utils.constants import API_KEY, DIAL_URL, DIAL_CHAT_COMPLETIONS_ENDPOINT
from task._utils.bucket_client import DialBucketClient
from task._utils.model_client import DialModelClient
from task._models.message import Message
from task._models.role import Role


async def _put_image() -> Attachment:
    file_name = 'dialx-banner.png'
    image_path = Path(__file__).parent.parent.parent / file_name
    mime_type_png = 'image/png'

    # 1. Create DialBucketClient
    async with DialBucketClient(api_key=API_KEY, base_url=DIAL_URL) as bucket_client:
        # 2. Open image file
        with open(image_path, 'rb') as image_file:
            # 3. Use BytesIO to load bytes of image
            image_bytes = BytesIO(image_file.read())

            # 4. Upload file with client
            response = await bucket_client.put_file(
                name=file_name,
                mime_type=mime_type_png,
                content=image_bytes
            )

            # 5. Return Attachment object with title (file name), url and type (mime type)
            return Attachment(
                title=file_name,
                url=response.get('url'),
                type=mime_type_png
            )


def start() -> None:
    # 1. Create DialModelClient
    client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name="gpt-4o",
        # You can try different models: gpt-4o, claude-3-5-sonnet-20240620, gemini-1.5-pro, etc.
        api_key=API_KEY
    )

    # 2. Upload image (use `_put_image` method)
    attachment = asyncio.run(_put_image())

    # 3. Print attachment to see result
    print("\n" + "=" * 80)
    print("Uploaded Image Attachment:")
    print("=" * 80)
    print(f"Title: {attachment.title}")
    print(f"URL: {attachment.url}")
    print(f"Type: {attachment.type}")
    print("=" * 80 + "\n")

    # 4. Call chat completion via client with list containing one Message:
    #    - role: Role.USER
    #    - content: "What do you see on this picture?"
    #    - custom_content: CustomContent(attachments=[attachment])
    message = Message(
        role=Role.USER,
        content="What do you see on this picture?",
        custom_content=CustomContent(attachments=[attachment])
    )

    print("Analyzing image...\n")
    response = client.get_completion(messages=[message])
    print(f"\nResponse: {response.content}\n")

    print("=" * 80)
    print("✅ Image analysis completed successfully!")
    print("Note: This approach uploads the image to DIAL bucket and references it via attachment.")
    print("The key benefit is that we can use Models from different vendors (OpenAI, Google, Anthropic).")
    print("The DIAL Core adapts this attachment to Message content in appropriate format for each Model.")
    print("TRY THIS APPROACH WITH DIFFERENT MODELS!")
    print("=" * 80 + "\n")


start()
