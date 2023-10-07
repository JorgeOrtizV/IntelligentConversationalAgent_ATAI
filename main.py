from speakeasypy import Speakeasy

if __name__ == "__main__":
    speakeasy = Speakeasy(host='https://speakeasy.ifi.uzh.ch', username='name', password='pass')
    speakeasy.login()

    rooms = speakeasy.get_rooms(active=True)

    for room in rooms:
        for message in room.get_messages(only_partner=True, only_new=True):
            # Implement your agent here #
            room.post_messages(f"Received your message: '{message.message}' ")
            room.mark_as_processed(message)

        for reaction in room.get_reactions(only_new=True):
            # Implement your agent here #
            room.post_messages(f"Received your reaction: '{reaction.type}' ")
            room.mark_as_processed(reaction)