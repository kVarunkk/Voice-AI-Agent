import re

def strip_markdown_for_speech(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)   # bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)        # italic
    text = re.sub(r'`(.*?)`', r'\1', text)          # inline code
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # headers
    text = re.sub(r'^[\-\*]\s+', '', text, flags=re.MULTILINE) # bullets
    return text