from telethon.sync import TelegramClient

api_id = 'TODO'
api_hash = 'TODO'

client = TelegramClient('test_session', api_id, api_hash)
client.start()
print(dir(client))
for message in client.get_messages('ml_progress_bot', limit=10000):
    client.download_media(message)
