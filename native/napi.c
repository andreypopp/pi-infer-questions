#include <node_api.h>
#include <stdbool.h>
#include <stdlib.h>

extern void pq_start(const char *directory);
extern char *pq_questions(const char *markdown, bool trace, char **failure);

typedef struct {
  napi_async_work work;
  napi_deferred deferred;
  char *markdown;
  bool trace;
  char *json;
  char *failure;
} Job;

static void execute(napi_env env, void *data) {
  Job *job = data;
  job->json = pq_questions(job->markdown, job->trace, &job->failure);
}

static void complete(napi_env env, napi_status status, void *data) {
  Job *job = data;
  napi_value value;
  if (job->json) {
    napi_create_string_utf8(env, job->json, NAPI_AUTO_LENGTH, &value);
    napi_resolve_deferred(env, job->deferred, value);
  } else {
    napi_value message;
    napi_create_string_utf8(env, job->failure ? job->failure : "failed", NAPI_AUTO_LENGTH, &message);
    napi_create_error(env, NULL, message, &value);
    napi_reject_deferred(env, job->deferred, value);
  }
  napi_delete_async_work(env, job->work);
  free(job->markdown);
  free(job->json);
  free(job->failure);
  free(job);
}

static char *string_of(napi_env env, napi_value value) {
  size_t length;
  napi_get_value_string_utf8(env, value, NULL, 0, &length);
  char *buffer = malloc(length + 1);
  napi_get_value_string_utf8(env, value, buffer, length + 1, &length);
  return buffer;
}

static napi_value start(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value argv[1];
  napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
  char *directory = string_of(env, argv[0]);
  pq_start(directory);
  free(directory);
  return NULL;
}

static napi_value questions(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value argv[2], promise, name;
  napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
  Job *job = calloc(1, sizeof(Job));
  job->markdown = string_of(env, argv[0]);
  if (argc > 1) napi_get_value_bool(env, argv[1], &job->trace);
  napi_create_promise(env, &job->deferred, &promise);
  napi_create_string_utf8(env, "questions", NAPI_AUTO_LENGTH, &name);
  napi_create_async_work(env, NULL, name, execute, complete, job, &job->work);
  napi_queue_async_work(env, job->work);
  return promise;
}

NAPI_MODULE_INIT() {
  napi_value fn;
  napi_create_function(env, "start", NAPI_AUTO_LENGTH, start, NULL, &fn);
  napi_set_named_property(env, exports, "start", fn);
  napi_create_function(env, "questions", NAPI_AUTO_LENGTH, questions, NULL, &fn);
  napi_set_named_property(env, exports, "questions", fn);
  return exports;
}
