---
layout:     page
title:      Archive
permalink:  /archive/
---

Find an article within this site using search terms: 

<script>
  (function() {
    var cx = '012587250564323129862:w5iqfetffay';
    var gcse = document.createElement('script');
    gcse.type = 'text/javascript';
    gcse.async = true;
    gcse.src = (document.location.protocol == 'https:' ? 'https:' : 'http:') + '//cse.google.com/cse.js?cx=' + cx;
    var s = document.getElementsByTagName('script')[0];
    s.parentNode.insertBefore(gcse, s);
  })();
</script>
<gcse:search></gcse:search>

## All Blog Posts in Reverse Chronological Order

{% for post in site.posts %}
  {% if post.visible %}
  * {{ post.date | date_to_string }} &raquo; [ {{ post.title }} ]({{ post.url | prepend:site.baseurl }})
  {% endif %}
{% endfor %}
