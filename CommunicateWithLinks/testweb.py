from langchain_helper import WebConvo

webconvo = WebConvo()


print(webconvo.load_and_chunk_web(["https://saharareporters.com/"]))