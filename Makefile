OS := $(shell uname -s)
NODE_INCLUDE ?= $(shell dirname $$(command -v node))/../include/node
DATA ?= data
BUILD ?= build
SOURCES = $(wildcard native/*.swift)

NPM_NAME = @andreypopp/pi-infer-questions
VERSION = $(shell node -p "require('./package.json').version")

ifeq ($(OS),Darwin)
NPM_PLATFORM = darwin-arm64
TARGET = -target arm64-apple-macos15
PACK ?= $(DATA)/coreml-ane-pack
GOLDEN ?= $(DATA)/golden-pi.json
PACK_FILES = question.mlmodelc option.mlmodelc settings.json tokenizer.json
SWIFT_LINK = -Xlinker -undefined -Xlinker dynamic_lookup
SWIFT_DEPS =
CMARK_FLAGS = -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 -DCMAKE_OSX_ARCHITECTURES=arm64
else
NPM_PLATFORM = linux-$(if $(filter x86_64,$(shell uname -m)),x64,arm64)
TARGET =
PACK ?= $(DATA)/onnx
GOLDEN ?= $(DATA)/golden-onnx-$(ORT_ARCH).json
PACK_FILES = question.onnx option.onnx settings.json tokenizer.json
ORT_VERSION = 1.31.0
ORT_ARCH = $(if $(filter x86_64,$(shell uname -m)),x64,aarch64)
ORT_SHA256_x64 = cc5c72baf5ae5c8238a6841f0897227be2d02826b9cf98eaf02fdefaeeb03a57
ORT_SHA256_aarch64 = c5b8b3cca31f3d643a3b313b8f42e2d0f16a4fcefd02cda343d97bd5306afaab
ORT = $(BUILD)/onnxruntime-linux-$(ORT_ARCH)-$(ORT_VERSION)
SWIFT_LINK = -static-stdlib -I $(ORT)/include -L$(ORT)/lib -lonnxruntime -Xlinker --strip-all -Xlinker --exclude-libs -Xlinker ALL -Xlinker -rpath -Xlinker '$$ORIGIN'
SWIFT_DEPS = $(ORT)/lib/libonnxruntime.so
CMARK_FLAGS = -DCMAKE_POSITION_INDEPENDENT_CODE=ON
endif

CMAKE ?= cmake
CMARK_VERSION = 0.29.0.gfm.13
CMARK_SHA256 = 5abc61798ebd9de5660bc076443c07abad2b8d15dbc11094a3a79644b8ad243a
CMARK = $(BUILD)/cmark-gfm
SWIFT_LINK += -I $(CMARK)/include -L$(CMARK)/lib -lcmark-gfm-extensions -lcmark-gfm
SWIFT_DEPS += $(CMARK)/lib/libcmark-gfm.a

PKG = $(BUILD)/npm/$(NPM_PLATFORM)
PKG_NAME = $(NPM_NAME)-$(NPM_PLATFORM)
LINK = node_modules/$(PKG_NAME)
DIST = $(BUILD)/npm-dist
MAIN_TGZ = $(DIST)/andreypopp-pi-infer-questions-$(VERSION).tgz

.PHONY: all package smoke clean e2e npm npm-linux publish

all: package

package: $(BUILD)/questions.node $(PACK)/settings.json
	mkdir -p $(PKG)/models
	@cmp -s $(BUILD)/questions.node $(PKG)/questions.node || { rm -f $(PKG)/questions.node; cp $(BUILD)/questions.node $(PKG)/questions.node; }
	$(if $(ORT),cp $(ORT)/lib/libonnxruntime.so.1 $(PKG)/libonnxruntime.so.1)
	@for f in $(PACK_FILES); do \
	  diff -rq $(PACK)/$$f $(PKG)/models/$$f >/dev/null 2>&1 || { rm -rf $(PKG)/models/$$f; cp -R $(PACK)/$$f $(PKG)/models/$$f; echo "models: $$f"; }; \
	done
	cp $(GOLDEN) $(PKG)/golden.json
	cp LICENSE NOTICE $(PKG)/
	mkdir -p $(PKG)/licenses/cmark-gfm && cp $(BUILD)/cmark-gfm-src/COPYING $(PKG)/licenses/cmark-gfm/
	printf '\nThis package includes cmark-gfm (BSD-2-Clause and others). Its license is in licenses/cmark-gfm/.\n' >> $(PKG)/NOTICE
	$(if $(ORT),mkdir -p $(PKG)/licenses/onnxruntime && cp $(ORT)/LICENSE $(ORT)/ThirdPartyNotices.txt $(PKG)/licenses/onnxruntime/ \
	  && printf '\nThis package includes ONNX Runtime (MIT License). Its license and third-party notices are in licenses/onnxruntime/.\n' >> $(PKG)/NOTICE)
	node scripts/platform-package.mjs $(NPM_PLATFORM) $(PKG)
	mkdir -p $(dir $(LINK))
	ln -sfn $(abspath $(PKG)) $(LINK)

$(BUILD)/napi.o: native/napi.c
	@mkdir -p $(BUILD)
	clang $(TARGET) -Wall -Werror -O2 -fPIC -I$(NODE_INCLUDE) -c $< -o $@

$(ORT)/lib/libonnxruntime.so:
	mkdir -p $(BUILD)
	curl -fsSL -o $(BUILD)/onnxruntime.tgz https://github.com/microsoft/onnxruntime/releases/download/v$(ORT_VERSION)/onnxruntime-linux-$(ORT_ARCH)-$(ORT_VERSION).tgz
	echo "$(ORT_SHA256_$(ORT_ARCH))  $(BUILD)/onnxruntime.tgz" | sha256sum -c -
	tar -xzf $(BUILD)/onnxruntime.tgz -C $(BUILD)
	rm $(BUILD)/onnxruntime.tgz
	printf 'module OnnxRuntime {\n  header "onnxruntime_c_api.h"\n  export *\n}\n' > $(ORT)/include/module.modulemap

$(CMARK)/lib/libcmark-gfm.a:
	mkdir -p $(BUILD)/cmark-gfm-src
	curl -fsSL -o $(BUILD)/cmark-gfm.tgz https://github.com/github/cmark-gfm/archive/refs/tags/$(CMARK_VERSION).tar.gz
	echo "$(CMARK_SHA256)  $(BUILD)/cmark-gfm.tgz" | sha256sum -c -
	tar -xzf $(BUILD)/cmark-gfm.tgz -C $(BUILD)/cmark-gfm-src --strip-components=1
	$(CMAKE) -S $(BUILD)/cmark-gfm-src -B $(BUILD)/cmark-gfm-src/build -DCMAKE_BUILD_TYPE=Release \
	  $(CMARK_FLAGS) -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMARK_SHARED=OFF -DCMARK_TESTS=OFF
	$(CMAKE) --build $(BUILD)/cmark-gfm-src/build
	$(CMAKE) --install $(BUILD)/cmark-gfm-src/build --prefix $(CMARK)
	printf 'module Cmark [system] {\n  header "cmark-gfm.h"\n  header "cmark-gfm-extension_api.h"\n  header "cmark-gfm-core-extensions.h"\n  export *\n}\n' > $(CMARK)/include/module.modulemap

$(BUILD)/questions.node: $(SOURCES) $(BUILD)/napi.o $(SWIFT_DEPS)
	rm -f $@
	swiftc $(TARGET) -O -swift-version 5 -warnings-as-errors -module-name questions -emit-library \
	  $(SWIFT_LINK) -o $@ $(SOURCES) $(BUILD)/napi.o

$(PACK)/settings.json:
	rm -rf $(BUILD)/fetch
	mkdir -p $(BUILD)/fetch $(PACK)
	@npm pack $(PKG_NAME)@$(VERSION) --pack-destination $(BUILD)/fetch --silent >/dev/null \
	  || { echo "no models in $(PACK) and $(PKG_NAME)@$(VERSION) is not published: train and convert them (see the pipeline), or publish the package"; exit 1; }
	tar -xzf $(BUILD)/fetch/*.tgz -C $(PACK) --strip-components=2 package/models
	tar -xzf $(BUILD)/fetch/*.tgz -O package/golden.json > $(GOLDEN)

smoke: package
	node test/smoke.js $(PKG)/questions.node $(PKG)/models $(PKG)/golden.json

npm: package
	mkdir -p $(DIST)
	npm pack $(PKG) --pack-destination $(DIST)
	npm pack . --pack-destination $(DIST)

# On macOS: the Linux tarballs, built in podman containers (x64 runs emulated).
npm-linux:
	for arch in arm64 amd64; do \
	  podman build --platform linux/$$arch --build-arg TARGETARCH=$$arch -t pi-infer-questions-linux:$$arch linux || exit 1; \
	  podman run --rm --platform linux/$$arch -v $(CURDIR):/work pi-infer-questions-linux:$$arch \
	    make npm BUILD=build/linux-$$arch LINK=build/linux-$$arch/link DIST=$(DIST) || exit 1; \
	done

publish:
	@test -n "$$(ls $(DIST)/*.tgz 2>/dev/null)" || { echo "no tarballs in $(DIST): run make npm on every platform"; exit 1; }
	for t in $(filter-out $(MAIN_TGZ),$(wildcard $(DIST)/*.tgz)) $(MAIN_TGZ); do npm publish $$t --access public || exit 1; done

clean:
	rm -rf $(BUILD) node_modules

e2e: package
	node test/e2e.mjs extension.ts
