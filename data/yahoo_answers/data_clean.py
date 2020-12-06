from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re, htmlentitydefs


def unescape(text):
    """
    Removes HTML or XML character references and entities from a text string.
    @param text The HTML (or XML) source text.
    @return The plain text, as a Unicode string, if necessary.
    AUTHOR: Fredrik Lundh
    example:
        test_string = "&quot;The C Programming Laanguage&quot; is a nice book."
        assert unescape(test_string) == '"The C Programming Laanguage is a nice book."'
    """
    def fixup(m):
        text = m.group(0)
        if text[:2] == "&#":
            # character reference
            try:
                if text[:3] == "&#x":
                    return unichr(int(text[3:-1], 16))
                else:
                    return unichr(int(text[2:-1]))
            except ValueError:
                pass
        else:
            # named entity
            try:
                text = unichr(htmlentitydefs.name2codepoint[text[1:-1]])
            except KeyError:
                pass
        return text # leave as is
    return re.sub("&#?\w+;", fixup, text)


def remove_html_tag(text, filter_comment=False):
    """ 
    If closing filter comment flag, on simple pages, this code can be used to process out HTML comments,
    reducing page size and increasing rendering performance.
    But this code is expected to mess up when a comment contains other comments or HTML tags.
    """    
    if filter_comment:
        text = re.sub(r'<!--.*?-->', " ", text)
    text = re.sub(r"<.*?>", "", text)
    return text


def clean_ag_news(text):
    text = text.replace('\\n',' ')
    text = text.replace('\\',' ')
    text = re.sub(r'#\d{1,};',' ',text)
    text = unescape(text)
    text = remove_html_tag(text)
    text = re.sub(r's\{2,}',' ', text)
    return text
