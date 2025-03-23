import { useEffect, useState } from 'react';

export async function FetchData(endpoint: string, options: RequestInit = {}) {
    try {
        const response = await fetch(endpoint, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' },
            ...options,
        });

        if (!response.ok) {
            throw new Error(`Error: ${response.status}`);
        }
        const text = await response.text(); // Get raw response
        // console.log('Response Text:', text);

        return JSON.parse(text); // Manually parse JSON
    } catch (error) {
        console.error('API Call Failed:', error);
        throw error;
    }
};