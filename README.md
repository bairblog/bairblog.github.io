# BAIR Blog Instructions

This is the repository for the BAIR blog, located at `bair.berkeley.edu/blog/`.
Posts are located in `_posts`. Preview locally, and then push to the master
branch, which will generate a preview at `bairblog.github.io`. This is the link
we give to authors to preview their posts.

Contents:

- [Quick Pipeline](#quick-pipeline)
- [Writing Posts](#writing-posts)
    - [Images](#images)
    - [YouTube Videos](#youtube-videos)
    - [Previewing Locally](#previewing-locally)
- [How to Update](#how-to-update)
    - [Steps](#steps)
    - [Push to the Server](#push-to-the-server)
    - [Setting Permissions](#setting-permissions)
- [References](#references)


# Quick Pipeline

A quick TL;DR for the entire pipeline:

- Download all the images and GIFs that the author gave and copy over to the
  blog's `static` folder on the server, *and adjust permissions* (don't
  forget!).
- Pick one representative image for the blog post and add that to the GitHub (so
  it's in the `assets` folder).
- Copy and paste stuff from the Google Doc to a new post in the `_posts` folder.
  When writing, refer to figures on the blog server.
- **Copy the three lines of code that have the Twitter card**, taking care to
  ensure that the title and images change based on each post.
- Send the draft to the authors, get feedback, revise, etc.
- Publish live to the server by pushing to the `production` branch.  Adjust
  permissions of all new folders created.
- Send MailChimp campaign, being sure to adjust the links to the new blog post
  in the three designated areas.

See the following sections for details.


# Writing Posts

There are a few important things to know about our specific blog format. This
section is intended as a guide for members of the blog editorial board when they
have to convert posts in Google Doc form (from the student/postdoc/faculty
authors) to Markdown.

## Images

To avoid putting all images and gifs into this GitHub repository, and to avoid
having to copy the entire `_site` folder to where the blog lives, we save the
images inside the folder `/project/eecs/interact/www-bair/static/blog`. To upload
images to the server, see instructions [here](#push-to-the-server) and [here](#setting-permissions).
Each blog post gets its own folder of images/gifs, even if it has only one.

Make sure you avoid using the `site.url` and `site.baseurl` liquid tags. That
is, earlier we used the following code:

```
<p style="text-align:center;">
<img src="{{site.url}}{{site.baseurl}}/assets/mh_test/different_tests.png" alt="different_tests" width="600"><br>
<i>
Functions $f$ and $g$ can serve as acceptance tests for Metropolis-Hastings.
Given current sample $\theta$ and proposed sample $\theta'$, the vertical axis
represents the probability of accepting $\theta'$.
</i>
</p>
```

But **do not use that**. Instead, use an explicit link to the static folder
(and please use `https` instead of `http`):

```
<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/mh_test/different_tests.png" alt="different_tests" width="600"><br>
<i>
Functions $f$ and $g$ can serve as acceptance tests for Metropolis-Hastings.
Given current sample $\theta$ and proposed sample $\theta'$, the vertical axis
represents the probability of accepting $\theta'$.
</i>
</p>
```

The above also represents how we prefer to use captions for figures, and how to
center images, control width, etc. Note, however, that using hyperlinks in
Jekyll format (using `[text](link)`) for the captions doesn't work, so use the
explicit HTML code.

Set `width=""` for images that should span the full column width.

Remove the italics if there is no figure caption.


## YouTube Videos

To insert YouTube videos, use

```
{% include youtubePlayer.html id="yourVideoID" %}
```

For example, if the video is `https://www.youtube.com/watch?v=XPFQ9TBvtCE`,
then the ID would be `XPFQ9TBvtCE`.

## Previewing Locally

To preview locally, run `bundle exec jekyll serve` as described in [the Jekyll
starter guide][2]. If all goes well, and you [have jekyll installed][7], you
should see the following output:

```
danielseita$ bundle exec jekyll serve
Configuration file: /Users/danielseita/bairblog/_config.yml
Configuration file: /Users/danielseita/bairblog/_config.yml
            Source: /Users/danielseita/bairblog
       Destination: /Users/danielseita/bairblog/_site
 Incremental build: disabled. Enable with --incremental
     Generating...
                    done in 0.434 seconds.
 Auto-regeneration: enabled for '/Users/danielseita/bairblog'
Configuration file: /Users/danielseita/bairblog/_config.yml
    Server address: http://127.0.0.1:4000/blog/
  Server running... press ctrl-c to stop.
```

Note that this may not work for outdated Jekyll versions. I'm using version
3.4.4. here. Also, this would correspond to the production branch, not the
master (and for converting posts, you should generally be working with the
master branch).

Once you did that, copy the URL `http://127.0.0.1:4000/blog/` to your web
browser and you should see your post there. Double check that all links are
working and that all images are showing up. Oh, and if you modify your post and
save it, Jekyll automatically refreshes the website, so just refresh it in your
browser.



# How to Update

This section of the README explains the process of how to update the official
BAIR Blog at `bair.berkeley.edu/blog`. At this point, the person pushing it
should have gotten confirmation from the student authors that the preview on
`bairblog.github.io` looks good.

## Steps

- We have two branches, master and production. Use master for
  `bairblog.github.io` which is GitHub's built-in feature for updating websites,
  and for giving previews to authors.  The production branch is used for copying
  the files over to the place where the blog lives.

- When you're ready to deploy, do `git checkout production` and then `git merge
  master` to get the production branch updated. However ...

- ... that production branch has specific `baseurls` and `urls` that shouldn't
  change. This will be different from the master branch, so watch out!  The
  master branch uses:

  ```
  baseurl:     ""
  url:         "https://bairblog.github.io/"
  ```

  But the production branch needs to use:

  ```
  baseurl:     "/blog"
  url:         "http://bair.berkeley.edu"
  ```

  Don't forget to also deal with comments! See next bullet point.

- Set comments to be False (by setting `show_comments: False`) on the master
  branch, and during any development. Set it to be True only when we first
  publish it live. Disqus is tracking the first instance when it sees the
  comments section being generated, so these comment alert emails Disqus is
  sending redirects to posts under /jacky or /jane. (Note: this was before when
  we had separate folders for previewing the BAIR Blog.)

- Make posts invisible (by setting `visible: False`) until we are ready for them
  to be pushed live. This isn't a problem if it's the master branch, but if it's
  production, and we copy things over, then people will see future posts ...

- Once things have been [deployed to the server](#push-to-the-server), you're
  not done! You have to change permissions of any file you may have created. For
  this, check:

  - Any new directories created. Every new month, for instance, starts a new
    directory, and then the day within that month, etc., and all these need to
    have their permissions set to be more generic.
  - Any images in the `assets` and/or `static/blog` folders that you added.

  See the [corresponding subsection on setting permissions](#setting-permissions).

- For the next post, switch back to the master branch and merge production
  there. You will need to update `_config.yml` again. Unfortunately I don't know
  of a better way around it.

**TODO add instructions on how to manage potential merging issues?**


## Push to the Server

For previewing locally, you should use:

```
bundle exec jekyll serve
```

However, if you want to get this actually deployed online, you need to run it in
**production mode**:

```
JEKYLL_ENV=production bundle exec jekyll serve
```

This should *still* generate an appropriate blog for you to preview *locally* at
`http://127.0.0.1:4000/blog/`. However, the difference with this command is that
it will *automatically* configure the `_site` folder to have the correct links
in it (bair.berkeley.edu and whatever your baseurl was). If you didn't add the
production environment, the `_site` folder would contain a bunch of
`http://localhost:4000` links.

Then, copy over the files to the server:

```
scp -r _site/* seita@login.eecs.berkeley.edu:/project/eecs/interact/www-bair/blog/
```

so that the contents of `_site` go in `blog`.  **There should be no permission
denied errors here (including files not relevant to the current update). If
there are, then we made a mistake with permissions somewhere**.


## Setting Permissions

When you copy files over to where the blog lives, you must ensure that the
permissions are set appropriately so that other members of the blog editorial
board can edit the files, and that viewers can see it.

This requires changing the **group** to be `interact` and the permissions set to
775. Here's an example, assuming that I've created a folder called `confluence`
to put the corresponding blog post images/gifs here. After copying to the
folder, the permissions are correctly adjusted:

```
[seita@login:/project/eecs/interact/www-bair/static/blog]$ chgrp -R interact confluence
[seita@login:/project/eecs/interact/www-bair/static/blog]$ chmod -R 775 confluence
[seita@login:/project/eecs/interact/www-bair/static/blog]$ ls -lh confluence/
total 15736
-rwxrwxr-x   1 seita    interact    5.9M Dec 31 16:02 cityscapes_sample_results.gif
-rwxrwxr-x   1 seita    interact     94K Dec 31 16:02 csVis.png
-rwxrwxr-x   1 seita    interact     98K Dec 31 16:02 dpe8C7u.png
-rwxrwxr-x   1 seita    interact    316K Dec 31 16:02 NQaMdTl.png
-rwxrwxr-x   1 seita    interact    173K Dec 31 16:02 pascalVis.png
-rwxrwxr-x   1 seita    interact     65K Dec 31 16:02 pipeline.jpg
-rwxrwxr-x   1 seita    interact    126K Dec 31 16:02 sample_result.png
-rwxrwxr-x   1 seita    interact     71K Dec 31 16:02 sinha.png
-rwxrwxr-x   1 seita    interact    217K Dec 31 16:02 sNetColorVis.png
-rwxrwxr-x   1 seita    interact     91K Dec 31 16:02 sNetVis.png
-rwxrwxr-x   1 seita    interact    352K Dec 31 16:02 st1Ia9i.png
-rwxrwxr-x   1 seita    interact    107K Dec 31 16:02 teaser_h.jpg
```

The parent directory (`confluence`) should also have permissions set at `drwxrwxr-x`.

(By the way, you obviously need to be a member of the `interact` group to do
this ... so we need to contact the IT staff behind the server.)

Don't forget to also change permissions of the folders that start with
`year/month/day/title/...`.


# References

Important references for understanding Jekyll, particularly with regards to
links:

- [Jekyll's instructions][1], including [installation here][8].
- [Understanding baseurl][4]
- [Configuring for project GitHub pages][3] (this is a "project" page because we're
  putting it on bair.berkeley.edu/blog and not in a personal github website).
- [Changing the root URL to be the correct one][5]
- [Jekyll docs on configurations][6]

Installation note: if you're getting an error about not finding the gem
bundler, you may be able to fix it by [installing the correct bundler
version][9].


[1]:https://jekyllrb.com/docs/posts/
[2]:http://jekyllrb.com/docs/quickstart/
[2]:https://jekyllrb.com/docs/configuration/#specifying-a-jekyll-environment-at-build-time
[3]:http://downtothewire.io/2015/08/15/configuring-jekyll-for-user-and-project-github-pages/
[4]:https://byparker.com/blog/2014/clearing-up-confusion-around-baseurl/
[5]:https://github.com/jekyll/jekyll/issues/5853
[6]:https://jekyllrb.com/docs/configuration/#specifying-a-jekyll-environment-at-build-time
[7]:https://jekyllrb.com/docs/installation/
[8]:https://jekyllrb.com/docs/installation/
[9]:https://bundler.io/blog/2019/01/04/an-update-on-the-bundler-2-release.html
