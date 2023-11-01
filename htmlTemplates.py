css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #F3EAEA;
    color: #000;
}
.chat-message.bot {
    background-color: #F3EAEA;
    border: 2px solid #7D3C98;
    color: #000;
}
.chat-message .avatar {
  width: 5%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
       <!-- <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;"> -->
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <!-- <img src="https://i.ibb.co/rdZC7LZ/Photo-logo-1.png"> -->
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
