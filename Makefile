# BAIR Blog — build & test helpers
#
# Common flow:
#   make setup        # one-time: create the isolated Ruby/Bundler environment
#   make test         # build in prod mode + run checks (config URL, output, links)
#   make preflight    # colorful "ready to push to the server?" checklist (README)
#   make build-prod   # production build into ./_site
#
# All gems are installed into ./vendor/bundle (git-ignored) so this project has
# its own isolated environment and does not touch system/global gems.

SHELL        := /bin/bash

# Force a UTF-8 locale: Ruby-Sass (sass 3.7.x) raises "Invalid US-ASCII
# character" on UTF-8 content when the shell locale is unset/non-UTF-8.
export LANG   := en_US.UTF-8
export LC_ALL := en_US.UTF-8

SITE_DIR     := _site
BUNDLE_PATH  := vendor/bundle
CONFIG       := _config.yml
PROD_ENV     := JEKYLL_ENV=production

.DEFAULT_GOAL := help

.PHONY: help setup check-env build build-prod serve test validate check-url check-output check-links preflight clean

help: ## Show this help
	@echo "BAIR Blog make targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create the isolated Bundler environment and install gems
	bundle config set --local path '$(BUNDLE_PATH)'
	bundle install
	@echo "Environment ready in ./$(BUNDLE_PATH)"

check-env: ## Verify the Bundler environment exists and dependencies resolve
	@test -d $(BUNDLE_PATH) || { echo "ERROR: environment missing. Run 'make setup' first."; exit 1; }
	@bundle check >/dev/null 2>&1 || { echo "ERROR: gems out of date. Run 'make setup'."; exit 1; }
	@echo "OK: Bundler environment is present and satisfied."

build: check-env ## Development build into ./_site
	bundle exec jekyll build

build-prod: check-env ## Production build into ./_site (JEKYLL_ENV=production)
	$(PROD_ENV) bundle exec jekyll build
	@echo "OK: production build complete -> ./$(SITE_DIR)"

serve: check-env ## Local preview at http://localhost:4000 (production env)
	$(PROD_ENV) bundle exec jekyll serve

test: validate ## Build + run all checks (alias for validate)

validate: build-prod check-url check-output check-links ## Full build + checks
	@echo "OK: all checks passed."

check-url: check-env ## Verify site URL/config are valid (jekyll doctor + url sanity)
	@bundle exec jekyll doctor
	@url=$$(grep -E '^url:' $(CONFIG) | head -1 | sed -E 's/^url:[[:space:]]*//; s/["'\'']//g' | xargs); \
	if [ -z "$$url" ]; then echo "ERROR: 'url:' is empty in $(CONFIG)"; exit 1; fi; \
	if [[ "$$url" != http*://* ]]; then echo "ERROR: 'url:' ($$url) is not a valid absolute URL"; exit 1; fi; \
	echo "OK: site url = $$url"

check-output: ## Verify the production build produced the expected pages
	@test -f $(SITE_DIR)/index.html || { echo "ERROR: missing $(SITE_DIR)/index.html (run 'make build-prod')"; exit 1; }
	@test -f $(SITE_DIR)/feed.xml   || { echo "ERROR: missing $(SITE_DIR)/feed.xml"; exit 1; }
	@echo "OK: expected pages present (index.html, feed.xml)"
	@test -f $(SITE_DIR)/sitemap.xml && echo "OK: sitemap.xml present" || echo "NOTE: sitemap.xml not generated (not enabled for this site)"

check-links: ## Check internal links/images (uses html-proofer if available)
	@if bundle exec htmlproofer --version >/dev/null 2>&1; then \
		bundle exec htmlproofer $(SITE_DIR) --disable-external --allow-hash-href --no-enforce-https; \
		echo "OK: html-proofer link check passed."; \
	else \
		echo "SKIP: html-proofer not installed. To enable: gem install html-proofer (or add to Gemfile)."; \
	fi

preflight: build-prod ## Colorful "ready to push to the server?" checklist (from README.md)
	@pass=0; fail=0; warn=0; \
	G='\033[0;32m'; R='\033[0;31m'; Y='\033[1;33m'; B='\033[0;36m'; \
	M='\033[0;35m'; D='\033[2m'; BD='\033[1m'; UL='\033[4m'; N='\033[0m'; \
	FR=(⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏); \
	spin(){ for k in 0 1 2 3 4 5 6 7 8 9; do printf "\r  $${M}%s$${N}  $${D}%s$${N}" "$${FR[$$k]}" "$$1"; sleep 0.03; done; }; \
	ok(){   printf "\r  $${G}✓$${N}  %s\033[K\n" "$$1"; pass=$$((pass+1)); }; \
	bad(){  printf "\r  $${R}✗$${N}  $${R}%s$${N}\033[K\n" "$$1"; fail=$$((fail+1)); }; \
	wrn(){  printf "\r  $${Y}!$${N}  $${Y}%s$${N}\033[K\n" "$$1"; warn=$$((warn+1)); }; \
	inf(){  printf "  $${B}i$${N}  %s\n" "$$1"; }; \
	sub(){  printf "       $${D}└ %s$${N}\n" "$$1"; }; \
	sec(){  printf "\n  $${BD}$${UL}%s$${N}\n" "$$1"; }; \
	step(){ printf "\n  $${B}$${BD}%s.$${N} $${BD}%s$${N}\n" "$$1" "$$2"; }; \
	cmd(){  printf "        $${D}$$ $${N}$${G}%s$${N}\n" "$$1"; }; \
	note(){ printf "        $${D}%s$${N}\n" "$$1"; }; \
	printf '\n  '"$${M}"'✦'"$${N}"' '"$${BD}"'BAIR Blog'"$${N}"' '"$${D}"'·'"$${N}"' '"$${BD}"'pre-push checklist'"$${N}"' '"$${D}"'(README.md)'"$${N}"'\n'; \
	printf '  '"$${D}"'──────────────────────────────────────────────────'"$${N}"'\n'; \
	sec "Build & configuration"; \
	spin "checking production build output…"; \
	if [ -f $(SITE_DIR)/index.html ] && [ -f $(SITE_DIR)/feed.xml ]; then ok "Production build output present (_site/index.html, feed.xml)"; else bad "Build output missing — run 'make build-prod'"; fi; \
	url=$$(grep -E '^url:' $(CONFIG) | head -1 | sed -E 's/^url:[[:space:]]*//; s/["'\'']//g' | xargs); \
	base=$$(grep -E '^baseurl:' $(CONFIG) | head -1 | sed -E 's/^baseurl:[[:space:]]*//; s/["'\'']//g' | xargs); \
	spin "validating site url…"; \
	if [[ "$$url" == http*://* ]]; then ok "Site url is a valid absolute URL ($$url)"; else bad "Site 'url:' is missing/invalid in $(CONFIG)"; fi; \
	spin "checking config profile…"; \
	if [ "$$base" = "/blog" ] && [[ "$$url" == *bair.berkeley.edu* ]]; then \
		ok "config matches PRODUCTION profile (baseurl=/blog, url=$$url)"; \
	else \
		wrn "config is not the production profile (README > How to Update)"; \
		sub "server needs: baseurl=\"/blog\"  url=\"http://bair.berkeley.edu\""; \
		sub "current:      baseurl=\"$$base\"  url=\"$$url\""; \
	fi; \
	sec "Content hygiene"; \
	spin "scanning for {{ site.url/baseurl }} tags…"; \
	posts=$$(grep -rlE '\{\{[[:space:]]*site\.(url|baseurl)' _posts 2>/dev/null | sort); \
	n=$$(printf '%s' "$$posts" | grep -c . || true); \
	if [ "$$n" -eq 0 ]; then ok "No {{ site.url }}/{{ site.baseurl }} tags in _posts"; \
	else wrn "$$n post(s) use {{ site.url/baseurl }} — README says use explicit https links"; \
		for p in $$posts; do c=$$(grep -cE '\{\{[[:space:]]*site\.(url|baseurl)' "$$p"); sub "$$(basename $$p) ($$c tag(s))"; done; fi; \
	spin "scanning for insecure http:// static links…"; \
	posts=$$(grep -rlE 'http://bair\.berkeley\.edu/static' _posts 2>/dev/null | sort); \
	n=$$(printf '%s' "$$posts" | grep -c . || true); \
	if [ "$$n" -eq 0 ]; then ok "All bair.berkeley.edu/static image links use https"; \
	else wrn "$$n post(s) use insecure http:// static links — README says use https"; \
		for p in $$posts; do c=$$(grep -cE 'http://bair\.berkeley\.edu/static' "$$p"); sub "$$(basename $$p) ($$c link(s))"; done; fi; \
	spin "scanning for visible:false posts…"; \
	posts=$$(grep -rlE 'visible:[[:space:]]*[Ff]alse' _posts 2>/dev/null | sort); \
	n=$$(printf '%s' "$$posts" | grep -c . || true); \
	if [ "$$n" -eq 0 ]; then ok "No posts marked visible:false"; \
	else wrn "$$n post(s) have visible:false — they will NOT publish"; \
		for p in $$posts; do sub "$$(basename $$p)"; done; fi; \
	spin "checking newest post Twitter card…"; \
	newest=$$(ls -1 _posts/*.md | sort | tail -1); \
	if grep -q 'twitter:title' "$$newest"; then ok "Newest post has a Twitter card ($$(basename $$newest))"; else wrn "Newest post ($$(basename $$newest)) is missing the Twitter card meta lines — README"; fi; \
	sec "Image links (newest post)"; \
	spin "resolving newest post images…"; \
	curlok=$$(command -v curl >/dev/null 2>&1 && echo 1 || echo 0); \
	bn=$$(basename "$$newest" .md); \
	y=$${bn:0:4}; mo=$${bn:5:2}; dd=$${bn:8:2}; slug=$${bn:11}; \
	html="$(SITE_DIR)/$$y/$$mo/$$dd/$$slug/index.html"; \
	[ -f "$$html" ] || html=$$(find $(SITE_DIR) -path "*$$slug*" -name index.html 2>/dev/null | head -1); \
	if [ -z "$$html" ] || [ ! -f "$$html" ]; then wrn "Could not locate built HTML for $$bn to verify images"; \
	else \
		imgs=$$(grep -oE '<img[^>]+src="[^"]+"' "$$html" | sed -E 's/.*src="([^"]+)".*/\1/' | sort -u); \
		tot=0; bl=0; br=0; BADL=""; BADR=""; \
		for src in $$imgs; do \
			tot=$$((tot+1)); \
			case "$$src" in \
				http://*|https://*) \
					if [ "$$curlok" = "1" ]; then \
						spin "checking remote image $$tot…"; \
						code=$$(curl -sSL -o /dev/null -w '%{http_code}' --max-time 15 "$$src" 2>/dev/null || echo 000); \
						if [ "$$code" -ge 400 ] || [ "$$code" = "000" ]; then br=$$((br+1)); BADR="$$BADR\n$$code  $$src"; fi; \
					fi; \
					;; \
				*) \
					p=$${src%%\#*}; p=$${p%%\?*}; p=$${p#/blog}; p=$${p#/}; \
					if [ ! -f "$(SITE_DIR)/$$p" ] && [ ! -f "$$p" ]; then bl=$$((bl+1)); BADL="$$BADL\n$$src"; fi; \
					;; \
			esac; \
		done; \
		if [ "$$tot" -eq 0 ]; then ok "Newest post ($$bn) has no <img> tags to check"; \
		elif [ "$$bl" -eq 0 ] && [ "$$br" -eq 0 ]; then \
			if [ "$$curlok" = "1" ]; then ok "All $$tot image link(s) OK — local files exist, remote URLs reachable"; \
			else ok "All $$tot local image link(s) exist (curl missing: remote not HTTP-checked)"; fi; \
		else \
			wrn "Newest post images: $$bl missing local, $$br unreachable remote (of $$tot)"; \
			[ "$$curlok" = "1" ] || sub "curl not found — remote URLs were not HTTP-verified"; \
			printf '%b' "$$BADL" | while IFS= read -r ln; do [ -n "$$ln" ] && sub "missing local: $$ln"; done; \
			printf '%b' "$$BADR" | while IFS= read -r ln; do [ -n "$$ln" ] && sub "unreachable: $$ln"; done; \
		fi; \
	fi; \
	sec "Manual deploy steps (copy-paste, in order)"; \
	bn=$$(basename "$$newest" .md); slug=$${bn:11}; \
	imgdir=$$(grep -oE 'static/blog/[^/"]+' "$$newest" 2>/dev/null | head -1 | sed -E 's#static/blog/##'); \
	[ -n "$$imgdir" ] || imgdir="$$slug"; \
	SRV=seita@login.eecs.berkeley.edu; \
	note "Deploy user/host below is '$$SRV' (from README) — change if yours differs."; \
	step 1 "Confirm the authors approved the preview on bairblog.github.io"; \
	note "Do not proceed until the student authors have signed off on the master-branch preview."; \
	step 2 "Switch to the production branch and pull in your changes"; \
	cmd "git checkout production"; \
	cmd "git merge master"; \
	step 3 "Set the PRODUCTION profile in _config.yml (production branch only)"; \
	note "Edit $(CONFIG) so it reads exactly:"; \
	note "  baseurl:     \"/blog\""; \
	note "  url:         \"http://bair.berkeley.edu\""; \
	note "Also set 'show_comments: True' on the post only when publishing live."; \
	step 4 "Build the site in production mode"; \
	cmd "make build-prod"; \
	note "(equivalent to: JEKYLL_ENV=production bundle exec jekyll build)"; \
	step 5 "Upload this post's images to the server's static folder"; \
	note "Local images live in assets/$$imgdir/ — they go in static/blog/$$imgdir/ on the server."; \
	cmd "ssh $$SRV 'mkdir -p /project/eecs/bair/www-bair/static/blog/$$imgdir'"; \
	cmd "scp -r assets/$$imgdir/* $$SRV:/project/eecs/bair/www-bair/static/blog/$$imgdir/"; \
	step 6 "Copy the built site into the live blog directory"; \
	cmd "scp -r _site/* $$SRV:/project/eecs/bair/www-bair/blog/"; \
	note "There should be NO 'permission denied' errors. If there are, permissions are wrong (see step 7)."; \
	step 7 "Fix group + permissions on everything you just uploaded"; \
	cmd "ssh $$SRV"; \
	cmd "cd /project/eecs/bair/www-bair/static/blog"; \
	cmd "chgrp -R bair-www $$imgdir && chmod -R 775 $$imgdir"; \
	note "Then do the same for the new dated post folders under .../blog (year/month/day/title):"; \
	cmd "cd /project/eecs/bair/www-bair/blog"; \
	cmd "chgrp -R bair-www $$(date +%Y) && chmod -R 775 $$(date +%Y)"; \
	note "You must be in the 'bair-www' group for this to work (contact IT if not)."; \
	step 8 "Switch back to master so the next edit starts clean"; \
	cmd "git checkout master"; \
	cmd "git merge production"; \
	note "Restore the master profile in $(CONFIG): baseurl: \"\"  url: \"https://bairblog.github.io/\""; \
	step 9 "Announce the post"; \
	note "Send the MailChimp campaign to the blog mailing list."; \
	printf '\n  '"$${D}"'──────────────────────────────────────────────────'"$${N}"'\n'; \
	printf "  $${D}summary:$${N} $${G}%d passed$${N} · $${Y}%d warning(s)$${N} · $${R}%d failed$${N}\n" "$$pass" "$$warn" "$$fail"; \
	if [ $$fail -gt 0 ]; then printf "\n  $${R}$${BD}✗ NOT ready to push — resolve the failures above.$${N}\n\n"; exit 1; \
	elif [ $$warn -gt 0 ]; then printf "\n  $${Y}$${BD}! Buildable — review the %d warning(s) before pushing.$${N}\n\n" "$$warn"; \
	else printf "\n  $${G}$${BD}✓ All checks passed — ready to push.$${N}\n\n"; fi

clean: ## Remove build output and caches
	rm -rf $(SITE_DIR) .jekyll-cache .jekyll-metadata .sass-cache
	@echo "OK: cleaned build artifacts."
