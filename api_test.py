from openai import OpenAI
client = OpenAI(
    base_url='https://xiaoai.plus/v1',
    api_key='sk-qBt8y4fvpCYAEMIVK09dQ760m5L6ONf79gGYpV5rDlYqqL12'
)
completion = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "system", "content": "you are a helpful assistant"},
    {"role": "user", "content": "do you know what duck in the chinese mean?"}
  ]
)
print(completion.choices[0].message)