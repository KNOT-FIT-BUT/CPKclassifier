#!/usr/bin/env python3
"""
@ name          fulltext_parser.py
@ old_name		converter.py
@ author        xkurak00, xormos00
@ date 			June 2016
@ last_update   June 2017
@ license       VUT FIT UPGM
@ project       NAKI CPK
@ description   Skript pre spracovanie fulltextov z XML suborov do formatu vertikalizacie
"""

import os
import sys
from multiprocessing import Process, Queue
from subprocess import call
import re
re.U

# Vytvorenie directory a ulozenie suborov
def make_archives_queue(directory, queue):
	for arch in os.listdir(directory) :
		if arch.endswith(".tar.gz"):
			queue.put(arch)
# --------------------------
# Worker
def worker(input_dir, output_dir, queue, error_queue, num):
	tmp_dir = output_dir + "tmp" + str(num) + "/"
	if not os.path.exists(tmp_dir):
		os.makedirs(tmp_dir)
	while not queue.empty():
		file_name = queue.get()
		convert_file(input_dir, output_dir, tmp_dir, file_name, error_queue)


#--------------------------------------------------
def error_writer(queue, error_queue):
	with open("error_log.txt",'w') as output_file:
		while not queue.empty():
			if not error_queue.empty():
				output_file.write(error_queue.get())
				output_file.flush()


#-- Spracovanie suboru
def convert_file(input_dir, output_dir, tmp_dir, file_name, error_queue):
	# Priprava Error stringu
	print("Processing: ",file_name)
	error_string = ""
	# Extrakcia suboru
	command = "tar -zxvf "+ input_dir + file_name + " -C " + tmp_dir + " >/dev/null"
	os.system(command)
	# Spracovanie hlavneho xml
	file_name = "uuid:" + file_name.replace("tar.gz","xml")
	input_file_name = tmp_dir + file_name
	final = {}
	with open(input_file_name) as input_file:
		xml_text = input_file.read()
		final["title"] = re.findall("(?<=<dc:title>).*?(?=<\/dc:title>)", xml_text)
		final["creator"] = re.findall("(?<=<dc:creator>).*?(?=<\/dc:creator>)", xml_text)
		final["subject"] = re.findall("(?<=<dc:subject>).*?(?=<\/dc:subject>)", xml_text)
		final["publisher"] = re.findall("(?<=<dc:publisher>).*?(?=<\/dc:publisher>)", xml_text)
		final["date"] = re.findall("(?<=<dc:date>).*?(?=<\/dc:date>)", xml_text)
		final["identifier"] = re.findall("(?<=<dc:identifier>).*?(?=<\/dc:identifier>)", xml_text)
		final["language"] = re.findall("(?<=<dc:language>).*?(?=<\/dc:language>)", xml_text)
		final["rights"] = re.findall("(?<=<dc:rights>).*?(?=<\/dc:rights>)", xml_text)
		final["data_source"] = re.findall("(?<=rdf:resource=\"info:fedora/)uuid[^\"]*", xml_text)
		final["text"] = []
		meta_doc = "<meta_doc>"
		for name in final:
			if name != "data_source" and name != "text":
				if isinstance(final[name], list):
					for thing in final[name]:
						meta_doc += " <" + name + ">" + thing  + "</" + name + ">"
				else:
					meta_doc += " <" + name + ">" + final[name]  + "</" + name + ">"
		final["text"].append(["<doc>"])
		meta_doc += " </meta_doc>"
		final["text"].append([meta_doc])
		i = 0
		for record in final["data_source"]:
			page_file_name = tmp_dir + file_name.replace(".xml","") + "/" + record + ".xml"
			final["text"].append(["<page>"])
			meta_page = ["<meta_page>" + " <num>" + str(i) + "</num>" + " <source>"+ record + "</source>" + " </meta_page>"]
			final["text"].append( meta_page )

			try:
				with open(page_file_name) as page_file:
					text_lines = re.findall("(?<=TextLine).*?(?=TextLine>)", page_file.read())
					for line in text_lines:
						final["text"].append(re.findall("(?<=CONTENT=\")[^\"]*",line))
				os.remove(page_file_name)
			except:
				print("Error mk page")
				error_string += file_name + " :error loading page: " + record + "\n"
				final["text"].append(["<error>"])

			final["text"].append(["</page>"])
			i +=1
		final["text"].append(["</doc>"])

	try:
		os.rmdir(input_file_name.replace(".xml",""))
	except:
		print("Error rm page")
		for i in os.listdir(input_file_name.replace(".xml","")):
			os.remove(input_file_name.replace(".xml","") + '/' + i)
			error_string += file_name + " :error file in directory: " + i + "\n"
		os.rmdir(input_file_name.replace(".xml",""))
	os.remove(input_file_name)

	# VYtvorenie txt suboru
	output_string = ""
	for array in final["text"]:
		i = -1
		index = len(array) - 1
		for word in array:
			i += 1
			if i != index or word != "172":
				output_string += word + '\n'
			else:
				output_string = output_string[:-1]
	output_file_name = output_dir + file_name.replace("xml","txt")
	with open(output_file_name, 'w') as output_file:
		output_file.write(output_string)
	error_queue.put(error_string)
#------------------------------


input_dir = os.path.abspath(sys.argv[1]) + "/"
output_dir = os.path.abspath(sys.argv[2]) + "/"
num_of_workers = int(sys.argv[3])

queue = Queue()
error_queue = Queue()

make_archives_queue(input_dir, queue)


err_writer = Process( target = error_writer, args =(queue, error_queue))
err_writer.start()
workers = []
for num in range(num_of_workers):
	workers.append(Process(target = worker, args = (input_dir, output_dir, queue, error_queue, num)))

for worker in workers:
	worker.start()

for worker in workers:
	worker.join()
err_writer.join()