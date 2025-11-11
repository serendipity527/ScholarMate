from langchain.agents import create_agent

def send_email(to: str, subject: str, body: str):
    """Send an email"""
    email = {
        "to": to,
        "subject": subject,
        "body": body
    }
    # ... email sending logic

    return f"Email sent to {to}"
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.messages import HumanMessage

model = ChatTongyi(
    streaming=True,
    api_key="sk-43070f4cd1074965a93a03d6d5333cd8",
)
agent = create_agent(
    model=model,
    tools=[send_email],
    system_prompt="You are an email assistant. Always use the send_email tool.",
)