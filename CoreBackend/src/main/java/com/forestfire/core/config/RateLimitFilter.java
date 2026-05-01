package com.forestfire.core.config;

import jakarta.servlet.*;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

@Component
public class RateLimitFilter implements Filter {

    // Simple in-memory rate limiting (for demonstration)
    // In production, use Redis and Resilience4j
    private final ConcurrentHashMap<String, AtomicInteger> requestCounts = new ConcurrentHashMap<>();
    private final int MAX_REQUESTS_PER_MINUTE = 100;
    private long windowStartTime = System.currentTimeMillis();

    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain)
            throws IOException, ServletException {
        
        HttpServletRequest req = (HttpServletRequest) request;
        HttpServletResponse res = (HttpServletResponse) response;
        String clientIp = req.getRemoteAddr();

        // Reset window every minute
        if (System.currentTimeMillis() - windowStartTime > 60000) {
            requestCounts.clear();
            windowStartTime = System.currentTimeMillis();
        }

        requestCounts.putIfAbsent(clientIp, new AtomicInteger(0));
        int requests = requestCounts.get(clientIp).incrementAndGet();

        if (requests > MAX_REQUESTS_PER_MINUTE) {
            res.setStatus(429); // Too Many Requests
            res.getWriter().write("Too Many Requests. Please try again later.");
            return;
        }

        chain.doFilter(request, response);
    }
}
