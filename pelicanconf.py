#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals
import os

AUTHOR = 'Dunya Oguz'
SITENAME = 'dunyaoguz.github.io'
SITEURL = 'https://dunyaoguz.github.io/my-blog'
ABOUTURL = 'https://dunyaoguz.github.io/my-blog/pages/about'
CONTACTURL = 'https://dunyaoguz.github.io/my-blog/pages/contact'
PATH = 'content'
STATIC_PATHS = ['images']
TIMEZONE = 'America/Toronto'
DEFAULT_LANG = 'en'
DEFAULT_PAGINATION = 10

DISPLAY_PAGES_ON_MENU = False
USE_FOLDER_AS_CATEGORY = True 
ARTICLE_PATHS = ['articles',]
PAGE_PATHS = ['pages',]
MENUITEMS = (
   ('About', '/pages/about.html'),
   ('Contact', '/pages/contact.html'),
)

# set to False for Production, True for Development
if os.environ.get('PELICAN_ENV') == 'DEV':
    RELATIVE_URLS = True
else:
    RELATIVE_URLS = False
