import speech_recognition as sr
import loaded as ld
import pyttsx3 as tx
engine=tx.init()
def getVoice():
   r=sr.Recognizer()
   with sr.Microphone() as source:
      audio=r.listen(source)
   try:
      return r.recognize(audio)
   except LookupError:
      return "Sorry couldn't understand what you said"
 
def replying():
   while True:
      spell=getVoice()
      if  spell== 'quit':
            break
      engine.say(ld.chat2(spell))
      engine.runAndWait()
         
'''
 if getVoice()=='quit':
         break
      else:
'''
