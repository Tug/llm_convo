from gevent import monkey

monkey.patch_all()

import logging
import argparse
import tempfile
import os
import sys
sys.path.append('/Users/tug/Work/airaifr/llm_convo')

from llm_convo.agents import OpenAIChat, TerminalInPrintOut
from llm_convo.conversation import run_conversation


def main(model):
    agent_a = OpenAIChat(
        system_prompt="You are a machine learning assistant. Answer the users questions about machine learning with short rhymes. Ask follow up questions when needed to help clarify their question.",
        init_phrase="Hello! Welcome to the Machine Learning hotline, how can I help?",
        model=model,
    )
    agent_b = TerminalInPrintOut()
    run_conversation(agent_a, agent_b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4")
    args = parser.parse_args()
    main(args.model)
