from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.messages import HumanMessage
import os


def send_email(to: str, subject: str, body: str):
    """Send an email"""
    email = {
        "to": to,
        "subject": subject,
        "body": body
    }
    # ... email sending logic

    return f"Email sent to {to}"


TONGYI_API_KEY = os.getenv("TONGYI_API_KEY")
if not TONGYI_API_KEY:
    raise ValueError("Missing TONGYI_API_KEY environment variable")

model = ChatTongyi(
    streaming=True,
    api_key=TONGYI_API_KEY,
)
agent = create_agent(
    model=model,
    tools=[send_email],
    system_prompt="You are an email assistant. Always use the send_email tool.",
)