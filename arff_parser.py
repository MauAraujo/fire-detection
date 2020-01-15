#!/usr/bin/env python

import sys
import re

names_file = str(sys.argv[1])
data_file =  str(sys.argv[2])

names = open(names_file, "r")
data = open(data_file, "r")

lines = names.readlines()
def title(line): return re.search("title.", line, flags=re.IGNORECASE)
def sources(line): return re.search("source.", line, flags=re.IGNORECASE)
def attributes(line): return re.search("attribute information", line, flags=re.IGNORECASE)
def missing(line): return re.search("missing", line, flags=re.IGNORECASE)

title = list(filter(title, lines))
sources = list(filter(sources, lines))
attributes = list(filter(attributes, lines))
missing_att =  list(filter(missing, lines))

print(title)
print(sources)
print(attributes)
print(missing_att)
