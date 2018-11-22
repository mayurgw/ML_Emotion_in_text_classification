import re

def removeIrrelvantInfo(df,column='content',replace_reference='',replace_webpage=''):
    regex_references = re.compile('@[a-zA-Z0-9]+')
    df[column] = df[column].str.replace(regex_references,replace_reference)
    regex_websites = re.compile('http://[www.]*[a-zA-Z0-9]+.[a-z]+/[a-zA-Z0-9//]*')
    df[column] = df[column].str.replace(regex_websites,replace_webpage)
    return df
