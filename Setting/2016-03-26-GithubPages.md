---
layout:  post
title:  "Github Pages"
date:	2016-03-26 20:54:43 +0800
categories: [GitHub Pages, Help, Settings]
---
# Setting Github pages

## Preparing to install Github pages Gem

### Installing Ruby
{%highlight Linux%}
sudo apt-add-repository ppa:brightbox/ruby-ng
sudo apt-get update
sudo apt-get install ruby2.3 ruby2.3-dev
{%endhighlight%}

### Installing Bundler
{%highlight Linux%}
gem install bundler
{%endhighlight%}

## Installing the Github Pages Gem
1. create a file name **Gemfile** in <username>.github.io and add
{%highlight Linux%}
source 'https://ruby.taobao.org/'
gem 'github-pages'
 {%endhighlight%}

2. install Github Pages Gem
{%highlight Linux%}
bundle exec jekyll build --safe
{%endhighlight%}
If the command throw a warning about missing gems, then run
{%highlight Linux%}
 bundle install
 {%endhighlight%}
 If throw nokogiri error
{%highlight Linux%}
 sudo gem install nokogiri -- --use-system-libraries
 {%endhighlight%}
 3.run command **jeklly serve** to create **_site** folder.

## Uninstalling all gems
{%highlight Linux%}
 for i in `gem list --no-versions`; do gem uninstall -aIx $i; done
 {%endhighlight%}