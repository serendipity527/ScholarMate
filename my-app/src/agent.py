from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage
import os
from enum import Enum
from typing import Dict, Any


class IntentType(Enum):
    # 意图类型
    NORMAL_CONVERSATION = "普通对话"  # 日常聊天、简单问答、非学术性讨论
    DEEP_RESEARCH = "深度研究"  # 需要深入分析、多角度探讨、复杂问题研究
    LITERATURE_REPORT = "文献汇报生成"  # 要求生成学术报告、文献综述、研究总结


def send_email(to: str, subject: str, body: str):
    """发送邮件"""
    email = {
        "to": to,
        "subject": subject,
        "body": body
    }
    # ... 邮件发送逻辑

    return f"邮件发送给 {to}"


def research_literature(query: str, depth: str = "基础"):
    """基于查询进行文献研究"""
    # 模拟文献研究
    return f"研究文献：{query}，深度：{depth}"


def generate_literature_report(topic: str, style: str = "学术"):
    """生成文献报告"""
    # 模拟报告生成
    return f"生成文献报告：{topic}，风格：{style}"


TONGYI_API_KEY = os.getenv("TONGYI_API_KEY")
if not TONGYI_API_KEY:
    raise ValueError("缺少 TONGYI_API_KEY 环境变量")

model = ChatTongyi(
    streaming=True,
    api_key=TONGYI_API_KEY,
)


class IntentRouter:
    def __init__(self, model):
        self.model = model
        self.intent_classifier_prompt = """
你是一个意图识别助手。请根据用户输入，识别其意图类型。

意图类型包括：
1. 普通对话
2. 深度研究
3. 文献汇报生成

请只返回意图类型，不要其他内容。

用户输入：{user_input}
"""
        
        self.agents = {
            IntentType.NORMAL_CONVERSATION: create_agent(
                model=model,
                tools=[send_email],
                system_prompt="你是一个友好的对话助手。可以进行日常聊天和简单问答。如果需要发送邮件，可以使用 send_email 工具。"
            ),
            IntentType.DEEP_RESEARCH: create_agent(
                model=model,
                tools=[send_email, research_literature],
                system_prompt="你是一个深度研究助手。能够进行深入分析、多角度探讨复杂问题。使用 research_literature 工具进行文献研究。"
            ),
            IntentType.LITERATURE_REPORT: create_agent(
                model=model,
                tools=[send_email, research_literature, generate_literature_report],
                system_prompt="你是一个学术报告生成助手。专门生成文献综述、学术报告和研究总结。使用 generate_literature_report 工具生成报告。"
            )
        }
    
    def classify_intent(self, user_input: str) -> IntentType:
        """识别用户意图"""
        prompt = self.intent_classifier_prompt.format(user_input=user_input)
        message = HumanMessage(content=prompt)
        response = self.model.invoke([message])
        
        intent_str = response.content.strip().lower()
        
        intent_mapping = {
            "普通对话": IntentType.NORMAL_CONVERSATION,
            "深度研究": IntentType.DEEP_RESEARCH,
            "文献汇报生成": IntentType.LITERATURE_REPORT
        }
        
        return intent_mapping.get(intent_str, IntentType.NORMAL_CONVERSATION)
    
    def route_and_process(self, user_input: str) -> str:
        """根据意图路由到相应的代理"""
        intent = self.classify_intent(user_input)
        
        print(f"识别意图: {intent.value}")
        
        agent = self.agents[intent]
        
        message = HumanMessage(content=user_input)
        response = agent.invoke({"messages": [message]})
        
        return response.messages[-1].content


# 创建意图路由代理
intent_router = IntentRouter(model)


def process_user_input(user_input: str) -> str:
    """主函数：处理用户输入（兼容旧版）"""
    return intent_router.route_and_process(user_input)


# 旧版代理（兼容旧版）
agent = create_agent(
    model=model,
    tools=[send_email, research_literature, generate_literature_report],
    system_prompt="你是一个智能助手，具备意图识别功能，可以进行普通对话、深度研究和文献汇报生成。",
)