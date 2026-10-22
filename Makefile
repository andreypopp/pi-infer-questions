OS := $(shell uname -s)
NODE_INCLUDE ?= $(shell dirname $$(command -v node))/../include/node
DATA ?= data
BUILD ?= build
MODELS ?= models
SOURCES = $(wildcard native/*.swift)

MODELS_VERSION ?= 0.1.0
RELEASE_URL ?= https://github.com/andreypopp/pi-infer-questions/releases/download/v$(MODELS_VERSION)

ifeq ($(OS),Darwin)
PLATFORM = macos-arm64
TARGET = -target arm64-apple-macos15
PACK ?= $(DATA)/coreml-ane-pack
GOLDEN ?= $(DATA)/golden-pi.json
PACK_FILES = question.mlmodelc option.mlmodelc settings.json tokenizer.json
SWIFT_LINK = -Xlinker -undefined -Xlinker dynamic_lookup
SWIFT_DEPS =
CMARK_FLAGS = -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 -DCMAKE_OSX_ARCHITECTURES=arm64
else
PLATFORM = linux-$(ORT_ARCH)
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

ARCHIVE = pi-infer-questions-models-$(PLATFORM).tar.gz

.PHONY: all models release-assets smoke install clean e2e

all: questions.node models

questions.node: $(BUILD)/questions.node
	rm -f $@
	cp $< $@
	$(if $(ORT),rm -f libonnxruntime.so* && cp -P $(BUILD)/libonnxruntime.so* .)

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
	$(if $(ORT),cp -P $(ORT)/lib/libonnxruntime.so* $(BUILD)/)

$(PACK)/settings.json:
	mkdir -p $(BUILD)/release $(PACK)
	@curl -fsSL -o $(BUILD)/release/$(ARCHIVE) $(RELEASE_URL)/$(ARCHIVE) \
	  && curl -fsSL -o $(BUILD)/release/$(ARCHIVE).sha256 $(RELEASE_URL)/$(ARCHIVE).sha256 \
	  || { echo "no models in $(PACK) and no release assets at $(RELEASE_URL)/$(ARCHIVE): train and convert them (see the pipeline), or set RELEASE_URL and MODELS_VERSION"; exit 1; }
	cd $(BUILD)/release && shasum -a 256 -c $(ARCHIVE).sha256
	tar -xzf $(BUILD)/release/$(ARCHIVE) -C $(PACK)
	mv $(PACK)/golden.json $(GOLDEN)

models: $(PACK)/settings.json
	@mkdir -p $(MODELS)
	@for f in $(PACK_FILES); do \
	  diff -rq $(PACK)/$$f $(MODELS)/$$f >/dev/null 2>&1 || { rm -rf $(MODELS)/$$f; cp -R $(PACK)/$$f $(MODELS)/$$f; echo "models: $$f"; }; \
	done

release-assets: models
	mkdir -p $(BUILD)/release
	cp $(GOLDEN) $(BUILD)/release/golden.json
	tar -czf $(BUILD)/release/$(ARCHIVE) -C $(PACK) $(PACK_FILES) -C $(abspath $(BUILD)/release) golden.json
	cd $(BUILD)/release && shasum -a 256 $(ARCHIVE) > $(ARCHIVE).sha256

smoke: all
	node test/smoke.js $(BUILD)/questions.node $(MODELS) $(GOLDEN)

install: all
	@test -n "$(DEST)" || { echo "DEST=<extension dir> is required"; exit 1; }
	mkdir -p $(DEST)
	cp extension.ts $(DEST)/extension.ts
	rm -f $(DEST)/questions.node
	cp $(BUILD)/questions.node $(DEST)/questions.node
	$(if $(ORT),rm -f $(DEST)/libonnxruntime.so* && cp -P $(BUILD)/libonnxruntime.so* $(DEST)/)
	@diff -rq $(MODELS) $(DEST)/models >/dev/null 2>&1 || { rm -rf $(DEST)/models; cp -R $(MODELS) $(DEST)/models; }

clean:
	rm -rf $(BUILD) $(MODELS) questions.node libonnxruntime.so*

e2e: DEST := $(BUILD)/extension
e2e: install
	node test/e2e.mjs $(DEST)/extension.ts
