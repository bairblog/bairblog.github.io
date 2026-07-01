---
layout: post
title: "Intelligence is Free, Now What? Data Systems for, of, and by Agents"
date: 2026-07-01 09:00:00
author: Aditya Parameswaran
img: https://bair.berkeley.edu/static/blog/intelligence-is-free-now-what/image6.png
excerpt_separator: <!--more-->
visible: True
show_comments: False
---

<!-- twitter -->
<meta name="twitter:title" content="Intelligence is Free, Now What? Data Systems for, of, and by Agents">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="https://bair.berkeley.edu/static/blog/intelligence-is-free-now-what/image6.png">

<meta name="keywords" content="data systems, AI agents, agentic workloads, LLM inference cost, agent memory, query optimization, agent swarms, database systems">
<meta name="description" content="As the cost of intelligence approaches zero, agents will become the dominant workload for data systems. We explore three challenges and opportunities: data systems for agents, of agents, and by agents.">
<meta name="author" content="Aditya Parameswaran">

<p style="text-align:center;">
<i>... government of the people, by the people, for the people ...</i><br>
&nbsp;&nbsp;&nbsp;&nbsp;&mdash; Abraham Lincoln, Gettysburg Address (1863)
</p>

<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/intelligence-is-free-now-what/image6.png" alt="A cartoon database character and an AI robot agent holding hands" width="450">
</p>

The cost of AI is dropping rapidly. GPT-4-class capabilities cost roughly <span class="tex2jax_ignore">\$30</span> per million tokens in early 2023; today the same runs under <span class="tex2jax_ignore">\$1</span>, and [some providers are pushing costs below <span class="tex2jax_ignore">\$0.10</span>](https://zuplo.com/learning-center/the-10x-cheaper-ai-era-api-pricing-strategy-obsolete). Across benchmarks, [inference prices have fallen between 9x and 900x per year](https://epochai.org/data-insights/llm-inference-price-trends), with a median decline near 50x. Even [frontier models are getting dramatically cheaper](https://tokenmix.ai/blog/ai-pricing-trends-history) each generation, with open-source models following closely behind. And crucially, even if "Nobel-Prize-winning genius-level" intelligence isn't here yet, the intelligence that suffices for the vast majority of knowledge work is here today, and getting cheaper by the month. **At this rate, we are soon entering the era of virtually free intelligence**&mdash;the kind that is more than enough for everyday knowledge work.

<!--more-->

So, what does this new era of near-free intelligence mean for data systems? We believe three new challenges&mdash;and opportunities&mdash;stem from near-zero inference costs:

**Data Systems *For* Agents.** Agents will soon become the dominant workload for data systems&mdash;with swarms of agents spun up in response to each end-user request. Given differences in characteristics between agents and humans&mdash;or applications acting on their behalf&mdash;*how should we redesign data systems for such agentic users?*

**Data Systems *Of* Agents.** As agents start taking on the bulk of knowledge work, a new substrate is needed for thousands of agents to manage state over long-running tasks, coordinate and reach consensus, and deal with failures. *What do data systems that reliably and efficiently run and manage agent swarms look like?*

**Data Systems *By* Agents.** Agents are rapidly becoming capable of synthesizing entire data systems in one go&mdash;meaning we can rebuild custom systems for each new workload. Verifying that such systems match intended behavior is a challenge. *What does it take to let agents synthesize data systems we can actually trust?*

<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/intelligence-is-free-now-what/image1.png" alt="A database character and an agent shaking hands, with labels 'of', 'for', and 'by'" width="450">
</p>

Next, we will discuss each in more detail, followed by discussing the intertwined future of data systems and agents, especially as the three challenges intersect.

## Data Systems For Agents

An agent querying a database doesn't behave like a person or a BI tool. It performs what we call [*agentic speculation*](https://arxiv.org/abs/2509.00997): a high-volume, heterogeneous stream of work spanning schema introspection, columnar exploration, partial and then full query formulation. With multiple agents each exploring portions of the hypothesis space, each user request could amount to 1000s of individual SQL queries. Now, users can issue 'high-level' data tasks, e.g., root-cause analysis&mdash;e.g., 'why did coffee sales in Berkeley drop this year'&mdash;or exploratory cohort analysis&mdash;e.g., 'which user segments are most likely to churn next quarter'&mdash;each involving a combinatorial space of potential joins, aggregations, and filter combinations.

<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/intelligence-is-free-now-what/image5.png" alt="An agent sending many SELECT SQL queries to a database and receiving results back" width="600">
</p>

The requests from these agents have various opportunities for optimization. For instance, on a text-to-SQL benchmark with multiple agents attempting each task, only 10-20% of the sub-plans are distinct. Thus, 80-90% of sub-queries perform duplicate work. The same experiments show task success rates significantly increasing with more agentic attempts&mdash;so the redundancy is actually helpful. But from the data system perspective it's wasted work.

An agent-first data system can exploit such properties to help agents make progress faster. It can reuse results across overlapping sub-plans, drawing on ideas from decades-old literature on [multi-query optimization](https://dl.acm.org/doi/10.1145/42201.42203) and [shared scans](https://www.vldb.org/conf/2007/papers/research/p723-zukowski.pdf). Or the data system can try to *satisfice*, returning approximate answers that are good enough for agents to make progress, leveraging work from [the](https://dl.acm.org/doi/10.1145/253260.253291) [AQP](https://dl.acm.org/doi/10.1145/2465351.2465355) [literature](https://dl.acm.org/doi/10.1561/1900000004)&mdash;or streaming the results of the final or intermediate operators to help agents decide if seeing the rest is necessary or helpful.

Another opportunity here is to rethink the query interface entirely: instead of agents issuing a single SQL query at a time, they could instead issue a batch of queries, each with its own approximation requirements. Since enumerating an exponential search space (as in the root cause or cohort analysis examples above) isn't a good use of agentic reasoning ability, perhaps data systems should support higher-level primitives rather than requiring agents to list each SQL query explicitly. One idea here is to draw on [DBT-style Jinja macros](https://docs.getdbt.com/docs/build/jinja-macros) to provide looping-based primitives for agents to interact with data systems.

<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/intelligence-is-free-now-what/image2.png" alt="A swarm of AI agents working at laptops" width="450">
</p>

A final opportunity here is to stop thinking of data systems as passive executors of queries; data systems could be [proactive](https://arxiv.org/abs/2502.13016), as they possess more grounding in data and system characteristics that agents may lack a priori&mdash;they could steer agents in different directions, provide results for related queries, and also provide performance-level feedback (e.g., instead of executing an expensive query, the system could first provide the agent a latency estimate). The reason we can do this now as opposed to the past is that an agent can accept any form of textual feedback and isn't expecting a strict SQL query result. In fact, the data system could also prepare both materialized and virtual views for an agent in advance, provided to the agent as part of context, as this may be cheaper or more effective than having an agent author or use them.

## Data Systems Of Agents

Previously, we focused on how agents interact with data systems. Now, we consider everything else agents need to keep working: where they live, how they remember, how they coordinate with each other, and how they deal with failures of each other. This *agentic substrate* is separate from the inference stack powering raw intelligence. However, the inference stack itself is being abstracted away through APIs (e.g., from OpenAI or Anthropic), or, for open-weight models, through [serving](https://github.com/vllm-project/vllm) [frameworks](https://github.com/sgl-project/sglang) that hide low-level details. So far, the agentic substrate has been managed through harnesses like [Claude Code](https://www.anthropic.com/claude-code) and [Codex](https://github.com/openai/codex), coupled with various mechanisms to [store](https://mem0.ai/) and [retrieve](https://www.letta.com/) memory.

First, on the memory front, the current wisdom is that [files](https://www.amplifypartners.com/blog-posts/file-systems-for-agents) [are all you need](https://lsvp.com/stories/filesystemsforagents/); agents write to unstructured markdown (MD) files, which can then be searched using grep, or via embedding-based retrieval. In fact, many argue that the solution to continual learning is having agents consume a lot (e.g., an entire codebase, slack, company wikis, ...) and then write their learnings into MD files, which are then retrieved selectively on demand. Indeed, file systems, bash scripting, and MD files are and will still be important for agents. However, at scale, when agents are doing the vast majority of knowledge work, this approach will no longer be effective.

Given limited context windows, retrieving all MD file fragments that may be relevant and stuffing it into the context will break down at some point. Even if context windows continue to grow, there are latency benefits to not put all information into context &mdash; and in many cases, e.g., when knowledge work involves interacting with large databases or code bases, it will be infeasible to serialize all relevant data into context.

<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/intelligence-is-free-now-what/image3.png" alt="An agent perched on a database, drawing memory and state from it through glowing connections" width="400">
</p>

One could use a [knowledge](https://mem0.ai/) [graph](https://www.getzep.com/) [representation](https://langchain-ai.github.io/langmem/), but knowledge graphs suffer from the same limitations as unstructured MD-based memory due to their lack of structured search. What one needs is to be able to retrieve only memory that is pertinent to the task, across multiple attributes (or facets) of interest. For example, an agent debugging a flaky test should be able to pull only the memories tagged with the relevant module, language, framework, and failure mode&mdash;rather retrieving based on keywords or embedding similarity. A separate issue is what to actually retrieve; raw agent traces with mistakes are not very useful as they will induce agents to repeat the same mistake&mdash;instead, we want the retrieved memory to be corrective.

We recently explored a related notion of [*structured memory*](https://arxiv.org/abs/2602.13521), where we organize memory across various attributes, each of which could be set as `*` to indicate universal applicability, or set as a list of values to be matched. For a data agent, the dimensions could include the columns and tables, type of operation, and finally, open-ended natural-language corrective instructions. So, we could include memory that only applies to a given type of operation (e.g., 'when performing date-time operations, use fiscal year as opposed to calendar year conventions'), or a given table (e.g., 'column product_cleaned is preferred over column product when querying on product name'). One open question is defining an *application-specific structured memory*&mdash;or what others have called [world models for memory](https://www.linkedin.com/feed/update/urn:li:activity:7467499112523804672/). We believe this is akin to defining a schema for each application&mdash;and perhaps agents themselves can help us define and refine it over time.

Structured memory will be useful also for [evolutionary](https://github.com/skydiscover-ai/skydiscover) [frameworks](https://arxiv.org/abs/2506.13131) to effectively manage search spaces. Indeed, storing, structuring, and mining large volumes of single and [multi-agent traces](https://sky.cs.berkeley.edu/project/mast/) can help future agents become much more efficient&mdash;potentially enabling effective recursive self-improvement through structured memory-based mechanisms.

Another challenge is to support concurrent edits to shared memory, and concurrent edits in general, when there are many agents performing transformations. While there have been some useful attempts at [supporting](https://dl.acm.org/doi/10.1145/3702634.3702955) [multiversioning](https://neon.com/docs/get-started/why-neon) and [copy-on-write semantics](https://docs.turso.tech/agentfs/introduction), it isn't clear that such techniques will suffice when thousands of agents are attempting to edit shared state at the same time. For instance, when agents are trying various potential transactions in response to a user request, the effects of the vast majority of these transactions need to be rolled back&mdash;with only the one 'correct' transaction's result persisting. Work on supporting exactly-once semantics is relevant here, as are underlying techniques based on CRDTs and operational transformation. For updates to fuzzy mechanisms such as memory, we may be able to sacrifice on consistency for perfect correctness in the interest of latency. While agents can reason about semantics to compensate or roll back their actions to eventually finalize most tasks, the primary challenge lies in the degree to which they step on each other's toes during the process. An important failure mode to be avoided is a form of "livelock," where incessant compensating actions prevent any meaningful progress.

Beyond shared state, other concerns emerge when trying to support an army of agents, including what to do when agents fail, how agents should communicate with each other (directly or through intermediate shared state), and how we should deal with straggler agents. There have been some developments in supporting durable multi-agent execution, such as [Temporal](https://temporal.io/solutions/ai), but it remains to be seen if such solutions will apply at scale across thousands of agents. On the topic of communication, we need mechanisms to enable agents to negotiate with each other. Imagine four developer agents attempting to reach consensus on a shared schema, with distinct but overlapping objectives. In a human setting, this would involve iterative discussion and compromise; for agentic swarms, we must define the mechanisms that allow them to converge on a design that reflects the underlying goals of their respective principals. Or if agents are all requiring access to a limited resource, again communication will be necessary. It remains to be seen if this is best done via centralized coordination, or if a decentralized approach is necessary.

## Data Systems By Agents

Finally, if intelligence is effectively free, then we can employ this intelligence to synthesize new data systems from scratch. Indeed, in many settings, general-purpose data systems may be overkill, as they have to support every schema, query, and hardware target. Given a workload, recent work, including [Bespoke OLAP](https://arxiv.org/abs/2603.02001) and [GenDB](https://arxiv.org/abs/2603.02081), has shown that one can use an agentic pipeline to synthesize a complete, workload-specific analytical engine&mdash;in minutes to a few hours, at a cost of a few dollars. The engines are disposable: when the workload shifts, one can simply regenerate them. Analogously, our work has shown that one can synthesize custom [key-value stores](https://arxiv.org/abs/2605.24096) from scratch, targeted to the workload. In fact, modern IDEs, such as [Kiro](https://kiro.dev/), elevate specifications for systems development to be a first-class citizen.

<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/intelligence-is-free-now-what/image4.png" alt="An agent chiseling a database character out of a block of stone" width="500">
</p>

The main issue, however, is that specifications are typically imperfect, and don't cover all corner cases. Present-day agents will exploit the missing specifications to reward-hack their way to a high performance metric. In our custom key-value store work, we found that one way to alleviate this is to have auxiliary verification agents trying to generate test cases that catch the exploitation of corner cases, essentially expanding the specification. Yet another approach is to both generate a system and a proof for its correctness together, for which we have found some [early success](https://arxiv.org/abs/2605.23109), but more needs to be done to solidify the approach. Further, it remains to be seen what is the best way to solicit human-written specifications for a system&mdash;can this be done in an iterative, human-in-the-loop manner, as opposed to a one-shot, incomplete one. Indeed, human-written specifications are incomplete even for manually authored software, so one would expect that future agents that are more aligned will increasingly exercise better judgement when making design decisions.

Other questions here involve testing whether starting from a mature system (e.g., Postgres) and removing components/functionality can lead to higher performance or more user trust. Separately, is there an opportunity to make the design composable, comprising various verified components that are mixed and matched given a workload? For example, perhaps the workload hasn't changed enough for the storage layer to be updated, but perhaps the query optimizer requires changes. A perhaps more viable proposition involves employing agents coupled with proof systems to target critical parts of the code associated with formal proofs, rather than doing so for the entire system.

A final opportunity here is to move away from the traditional data systems stack with clearly-defined interfaces (e.g., parser, query optimizer, storage manager, &hellip;) &mdash; that were each largely the prerogative of a single human team to manage. Instead, agents can find new ways to "blend" these components together, perhaps identifying new optimization opportunities as a result. Agents can also fill in missing gaps in functionality to make existing systems much more feature-complete, or reach feature-parity with other competing systems&mdash;or analogously, continuously refining open-source systems in response to feature requests or issues (perhaps filed by other agents!) Doing so in a way that prioritizes correctness, long-term maintenance, and human interpretability will be a challenge.

## Looking Further Ahead

In the era of near-free intelligence, data systems matter more than ever. As agents take on the bulk of knowledge work, the workload for data systems will change, the substrate they need to run on will have to be built, and increasingly, they will participate in designing data systems themselves. Each of these shifts opens up a new, exciting research agenda.

<p style="text-align:center;">
<img src="https://bair.berkeley.edu/static/blog/intelligence-is-free-now-what/image7.png" alt="A yin-yang symbol formed by a database and a robot agent" width="250">
</p>

Looking further out, the boundaries between agents and data systems will likely start to blur. For instance, agents may design the data systems they themselves run on, defining both the interfaces as well as the system components underneath. Both the interfaces and internals can be evolved over time by agents in a form of recursive self-improvement. There is also an opportunity to rethink data systems as a holistic source of truth for the entirety of relevant state: including raw data, memory, and coordination state, further erasing the distinctions between the data that is being queried by agents and data generated as a result of agentic activity. Finally, data systems may themselves incorporate agentic components, fundamentally evolving from passive computation engines into intelligent, proactive, self-optimizing architectures. It is hard to predict what the future may hold. We're in for a wild ride!
