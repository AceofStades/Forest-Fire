package com.forestfire.core.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import jakarta.servlet.http.HttpServletRequest;
import java.util.Enumeration;
import java.util.Map;

@RestController
@RequestMapping("/api/ml")
public class ProxyController {

    @Value("${INFERENCE_ENGINE_URL:http://localhost:8000}")
    private String inferenceEngineUrl;

    @Autowired
    private RestTemplate restTemplate;

    @RequestMapping(value = "/**", method = {RequestMethod.GET, RequestMethod.POST, RequestMethod.PUT, RequestMethod.DELETE})
    public ResponseEntity<String> proxyRequest(HttpServletRequest request, @RequestBody(required = false) String body) {
        String path = request.getRequestURI().substring("/api/ml".length());
        String url = inferenceEngineUrl + path;
        
        if (request.getQueryString() != null) {
            url += "?" + request.getQueryString();
        }

        HttpHeaders headers = new HttpHeaders();
        Enumeration<String> headerNames = request.getHeaderNames();
        while (headerNames.hasMoreElements()) {
            String headerName = headerNames.nextElement();
            if (!headerName.equalsIgnoreCase("host")) { // avoid host header conflicts
                headers.add(headerName, request.getHeader(headerName));
            }
        }

        HttpEntity<String> httpEntity = new HttpEntity<>(body, headers);
        
        try {
            return restTemplate.exchange(url, HttpMethod.valueOf(request.getMethod()), httpEntity, String.class);
        } catch (Exception e) {
            return ResponseEntity.status(500).body("Error forwarding request to inference engine: " + e.getMessage());
        }
    }
}
