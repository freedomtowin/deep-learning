import datetime
import re
import pandas as pd
import time
import sqlite3 as sq
from subprocess import Popen, PIPE
from slpp import slpp as lua
import configparser

def create_database(name):
    sql_data = "working.sqlite" #- Creates DB names SQLite
    conn = sq.connect(sql_data)
    return conn

def exec_applescript(script):
	p = Popen(['osascript', '-e', script])
	
config = configparser.ConfigParser()
config.read('dl.cfg')

class ProcessChat():
	
	def process(self,df):
	
		def banned(x):
			names= []
			for n in names:
				if x.lower()==n.lower():
					return 1
			return 0

		def advertisments(x):
			words = ['wts','wtb','guild', 'for sale', 'lf ',
					 'm+15', 'gold only', 'goldonly', 'mythic']
			for w in words:
				if w in x.lower():
					return 1
			return 0


		def guilds(x):
			match = re.search('<(.*)>',x) 
			if match is not None:
				return 1
			return 0
	
		df['guild'] = df.text.apply(lambda x: guilds(x))
		df['ad'] = df.text.apply(lambda x: advertisments(x))
		df['ban'] = df.player_name.apply(lambda x: banned(x))
		df['player'] = df.player_name.apply(lambda x: x.split('-')[0])

		df['text_len'] = df.text.apply(len)
		df['word_cnt'] = df.text.apply(lambda x: len(x.split(' ')))

		df['txt_msg'] = df.apply(lambda x: '<b>'+x['player']+'</b>: '+x['text'],axis=1)
		tmp = df[(df.guild==0)&(df.ad==0)&(df.ban==0)&(df.word_cnt>3.0)].copy()
		return tmp
	

class CollectData(ProcessChat):

	def __init__(self):
		self.character_name = config['WOW']['CHARACTER_NAME']
		self.realm = config['WOW']['REALM']
		self.character_indentifier = self.character_name + ' - ' + self.realm

		self.restart_wowc = """
		on is_running(appName)
			tell application "System Events" to (name of processes) contains appName
		end is_running

		set safRunning to is_running("World of Warcraft")

		if safRunning is true then
			try
				tell application "World of Warcraft" to quit
		
			end try
			delay 10.0
		end if

		"""#.format(**{'USERNAME':config['WOW']['USERNAME'],'PASSWORD':config['WOW']['PASSWORD']})
			
# 		set CR to ASCII character of 13
# 		set ENTR to 36
# 		set TB to 48

# 		tell application "System Events"
# 			tell application "World of Warcraft" to activate
# 			delay 5.0
# 			keystroke "{USERNAME}"
# 			delay 2.0
# 			key code TB
# 			delay 2.0
# 			keystroke "{PASSWORD}" & CR
# 			delay 8.0
# 			key code ENTR
# 			delay 10.0
# 			tell application "World of Warcraft" to activate
# 			key code ENTR
# 			keystroke "/chatlog" & CR
# 			delay 5.0
# 		end tell
			
		self.reload_wowc = """
			on is_running(appName)
				tell application "System Events" to (name of processes) contains appName
			end is_running

			set CR to ASCII character of 13
			set ENTR to 36
			set TB to 48

			set safRunning to is_running("World of Warcraft")

			if safRunning is true then
				try
					tell application "System Events"
						tell application "World of Warcraft" to activate
						key code ENTR
						keystroke "/reload" & CR
						delay 3.0
					end tell
				end try
			end if
		"""

        
		self.log_path = "../../../Applications/World of Warcraft/_retail_/Logs/WoWChatLog.txt"
		self.chat_path = f"../../../Applications/World of Warcraft/_retail_/WTF/Account/391294447#1/{self.realm}/{self.character_name}/SavedVariables/Elephant.lua"

	def savelogs(self):
		exec_applescript(self.restart_wowc)
		time.sleep(20)
		
		conn = create_database('working')
		cur = conn.cursor()
		
		with open(self.log_path ,'r', encoding="utf8") as file_handle:
			data = file_handle.readlines()
			

		PATTERN = '(\d+)/(\d+) (\d+):(\d+):(\d+).\d+  \[(.*)\] (.*?):(.*)'
		i=0
		df = pd.DataFrame(columns=('date','chat_name','player_name','text'))
		for line in data:
			match = re.search(PATTERN, line)
			if match is None:
				print(line)
				continue
			month = match.group(1)
			day = match.group(2)
			hour = match.group(3)
			minute = match.group(4)
			second = match.group(5)
	
			chat_name = match.group(6)
			player_name = match.group(7)
			text = match.group(8)
	
			try:
				date = datetime.datetime(year=datetime.datetime.now().year,month=int(month),day=int(day),hour=int(hour),minute=int(minute),second=int(second))
			except:
				print(month,day,hour,minute,second)
			df.loc[i,['date','chat_name','player_name','text']] = [date,chat_name,player_name,text]
			i+=1
			
		tmp = self.process(df)
		
		cur.execute('''drop table if exists trade_chat_stg''')

		trade_chat_stg = tmp
		trade_chat_stg.to_sql("trade_chat_stg", conn,index=False)
		
		qry = """
		select * from trade_chat_stg where date>(select max(date) from trade_chat_db)
		"""
		trade_chat_stg = pd.read_sql(qry,conn)
		trade_chat_stg.to_sql("trade_chat_db", conn, if_exists='append',index=False)
		tmp = pd.read_sql('select * from trade_chat_db',conn)
		return self.process(tmp)
		
	def collectlogs(self):
		conn = create_database('working')
		cur = conn.cursor()
		tmp = pd.read_sql('select * from trade_chat_db',conn)
		return self.process(tmp)
		
		
	def collectchat(self):
		exec_applescript(self.reload_wowc)
		time.sleep(20)
			
		with open(self.chat_path,'r', encoding="utf8") as file_handle:
			data = file_handle.read()
		result = []
		eleph = lua.decode('{'+data+'}')
		elp_log = eleph['ElephantDBPerChar']['char'][self.character_indentifier]['logs']['trade']['logs']
		for x in elp_log:
			if 'time' in x.keys() and 'prat' not in x.keys():
		
				result.append({'date':datetime.datetime.fromtimestamp(int(x['time'])),
							   'player_name':x['arg2'],
							   'text':x['arg1'],
							   'chat_name':'2. Trade'
							  })
		tmp = self.process(pd.DataFrame(result))
		return tmp
