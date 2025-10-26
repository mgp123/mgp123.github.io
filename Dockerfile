FROM ruby:2.7
RUN gem install bundler -v 2.4
WORKDIR /srv/jekyll
COPY Gemfile Gemfile.lock ./
RUN bundle install
#COPY . .
CMD ["jekyll", "serve", "--host", "0.0.0.0", "--livereload"]
#CMD ["bundle", "exec", "jekyll", "serve", "--host", "0.0.0.0"]
