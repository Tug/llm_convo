from gevent import monkey

monkey.patch_all()

import logging
import argparse
import tempfile
import os
import time
import sys
sys.path.append('/Users/tug/Work/airaifr/llm_convo')

from llm_convo.agents import MicrophoneInSpeakerTTSOut
from llm_convo.audio_input import get_whisper_model
from pyngrok import ngrok


def main():
    # model = get_whisper_model()
    mic_in_speaker = MicrophoneInSpeakerTTSOut()
    print('-> speak now')
    res = mic_in_speaker.get_response([])
    print(res)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
