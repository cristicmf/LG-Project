#!/usr/bin/env python
# Test openai library import

print("Testing OpenAI library import...")
try:
    from openai import OpenAI
    print("OpenAI library imported successfully!")
except ImportError as e:
    print("Import error:", str(e))
except Exception as e:
    print("Other error:", str(e))

print("Test completed.")
