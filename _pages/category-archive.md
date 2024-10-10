---
title: "Study"
layout: categories
permalink: /categories/
author_profile: true
sidebar_main: true
---

## Explore Topics by Category 📚

Welcome to the category archive! Here, you can explore various topics organized by category. Click on a category to dive into articles related to that topic.

---

{% assign categories = site.categories %}
<ul class="categories-list" style="list-style: none; padding: 0;">
  {% for category in categories %}
    <li style="margin-bottom: 20px;">
      <a href="{{ site.baseurl }}/categories/{{ category[0] | slugify }}/" style="font-size: 20px; text-decoration: none; color: #0066cc;">
        <i class="fas fa-folder-open" style="margin-right: 10px;"></i> {{ category[0] | capitalize }} 
        <span style="color: gray; font-size: 16px;">({{ category[1].size }} articles)</span>
      </a>
    </li>
  {% endfor %}
</ul>

---
