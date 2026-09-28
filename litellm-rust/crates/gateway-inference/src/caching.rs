use std::time::Duration;

use litellm_cache_response::{InferenceCache, InferenceCacheOptions};
use litellm_gateway_auth::AuthenticatedRequest;
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::{Error, Gateway};

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Controls {
    #[serde(rename = "no-cache")]
    no_cache: bool,
    #[serde(rename = "no-store")]
    no_store: bool,
    ttl: Option<f64>,
    #[serde(rename = "s-maxage", alias = "s-max-age")]
    max_age: Option<f64>,
}

type Prepared = (
    Map<String, Value>,
    Option<(InferenceCache, InferenceCacheOptions)>,
);

pub(crate) fn prepare(
    gateway: &Gateway,
    identity: &AuthenticatedRequest,
    body: Map<String, Value>,
) -> Result<Prepared, Error> {
    let controls: Controls = match body.get("cache").filter(|value| !value.is_null()) {
        Some(value) => serde_json::from_value(value.clone())
            .map_err(|error| Error::InvalidBody(error.to_string()))?,
        None => Controls::default(),
    };
    let caching: Option<bool> = body
        .get("caching")
        .filter(|value| !value.is_null())
        .map(|value| serde_json::from_value(value.clone()))
        .transpose()
        .map_err(|error| Error::InvalidBody(error.to_string()))?;
    let caller = identity.caller();
    let options = InferenceCacheOptions {
        caching,
        no_cache: controls.no_cache,
        no_store: controls.no_store,
        ttl: controls.ttl.map(duration).transpose()?,
        max_age: controls.max_age.map(duration).transpose()?,
        scope: serde_json::json!([
            caller.principal().authority(),
            caller.principal().subject(),
            caller.authentication().credential_id
        ])
        .to_string(),
    };
    Ok((
        body.into_iter()
            .filter(|(name, _)| !matches!(name.as_str(), "cache" | "caching"))
            .collect(),
        gateway.response_cache.clone().map(|cache| (cache, options)),
    ))
}

fn duration(seconds: f64) -> Result<Duration, Error> {
    Duration::try_from_secs_f64(seconds)
        .ok()
        .filter(|duration| !duration.is_zero())
        .ok_or_else(|| Error::InvalidBody("cache durations must be finite and positive".into()))
}

#[derive(Clone, Default)]
pub(crate) struct CacheHeaders(std::sync::Arc<std::sync::OnceLock<String>>);

impl litellm_host::hooks::RouteHooks<litellm_core::RouteError> for CacheHeaders {
    async fn before_provider_request(
        &self,
        wire: litellm_host::event::WireRequest,
        _: litellm_host::event::RequestContext,
    ) -> Result<litellm_host::event::WireRequest, litellm_core::RouteError> {
        Ok(wire)
    }

    async fn on_event(
        &self,
        event: litellm_host::event::MachineEvent,
    ) -> Result<(), litellm_core::RouteError> {
        if let litellm_host::event::MachineEvent::CacheHit { key } = event {
            let _ = self.0.set(key);
        }
        Ok(())
    }
}

impl CacheHeaders {
    pub(crate) fn apply(&self, mut response: axum::response::Response) -> axum::response::Response {
        if let Some(key) = self.0.get()
            && let Ok(value) = axum::http::HeaderValue::from_str(key)
        {
            response.headers_mut().insert("x-litellm-cache-key", value);
        }
        response
    }
}
