use std::{
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use litellm_cache::{Error, ExactCacheContext};
use serde_json::Value;

use crate::{
    CacheControls, CacheKeyField, CacheKeyInput, ExactResponseCache, ResponseCacheRequest,
};

#[derive(Clone)]
pub struct InferenceCache {
    backend: Arc<dyn ExactResponseCache>,
    namespace: String,
    max_entry_bytes: usize,
}

#[derive(Clone, Default)]
pub struct InferenceCacheOptions {
    pub caching: Option<bool>,
    pub no_cache: bool,
    pub no_store: bool,
    pub ttl: Option<Duration>,
    pub max_age: Option<Duration>,
    pub scope: String,
}

pub struct InferenceCacheSession {
    cache: InferenceCache,
    request: ResponseCacheRequest,
}

impl InferenceCache {
    pub fn new(
        backend: Arc<dyn ExactResponseCache>,
        namespace: String,
        max_entry_bytes: usize,
    ) -> Self {
        Self {
            backend,
            namespace,
            max_entry_bytes,
        }
    }

    pub fn max_entry_bytes(&self) -> usize {
        self.max_entry_bytes
    }

    pub fn session(
        &self,
        surface: &str,
        mut input: Value,
        options: InferenceCacheOptions,
    ) -> InferenceCacheSession {
        input.sort_all_objects();
        let key = CacheKeyInput {
            namespace: Some(format!("{}:inference-v2", self.namespace)),
            fields: [
                ("surface", surface.to_owned()),
                ("scope", options.scope),
                ("request", input.to_string()),
            ]
            .into_iter()
            .map(|(name, value)| CacheKeyField {
                name: name.into(),
                value: Some(value),
                api_parameter: true,
                internal_parameter: false,
            })
            .collect(),
            ..Default::default()
        };
        InferenceCacheSession {
            cache: self.clone(),
            request: ResponseCacheRequest {
                key,
                controls: CacheControls {
                    configured: true,
                    supported_call_type: true,
                    native_backend: true,
                    default_on: true,
                    caching: options.caching,
                    no_cache: options.no_cache,
                    no_store: options.no_store,
                    ..Default::default()
                },
                context: ExactCacheContext { ttl: options.ttl },
                max_age: options.max_age,
            },
        }
    }
}

impl InferenceCacheSession {
    pub fn key(&self) -> String {
        crate::cache_key(&self.request.key)
    }

    pub fn max_entry_bytes(&self) -> usize {
        self.cache.max_entry_bytes
    }

    pub fn writes(&self) -> bool {
        self.request.controls.writes()
    }

    pub async fn lookup(&self) -> Result<Option<Value>, Error> {
        self.cache.backend.async_lookup(&self.request, now()).await
    }

    pub async fn store(&self, value: Value) -> Result<(), Error> {
        if value.to_string().len() > self.cache.max_entry_bytes {
            return Ok(());
        }
        self.cache
            .backend
            .async_store(&self.request, value, now())
            .await
    }
}

fn now() -> Duration {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
}
