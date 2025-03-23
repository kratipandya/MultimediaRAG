import React, { useEffect, useState, useCallback } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import {
    Typography,
    Box,
    Divider,
    List,
    Card,
    CircularProgress
} from '@mui/material';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import ShuffleIcon from '@mui/icons-material/Shuffle';
import FormatAlignJustifyIcon from '@mui/icons-material/FormatAlignJustify';
import ResultArxivItem from './resultpage/ResultArxivItem';
import { FetchData } from '../services/RagApi';
import TypewriterText from './resultpage/TypeWriter';
import ResultItemVideo from './resultpage/ResultItemVideo';
import ImageDisplay from './resultpage/ImageDisplay';
import AudioDisplay from './resultpage/AudioDisplay';
import LoadingText from './resultpage/LoadingText';

const LOADING_MSG = [
    "Searching response over 2.5 million research papers...",
    "Searching through various multimedia sources...",
    "Searching corresponding FAISS indices...",
];

const RAG_LOADING_MSG = [
    "Querying over 2.5 million research articles...",
    "Deepseek R1 thinking about your query...",
    "Generating responses based on content below...",
    "Generating answer based on research articles...",
];

// Polling interval in milliseconds
const POLLING_INTERVAL = 10 * 1000; // Poll every 10 seconds

function ResultPage() {
    const [searchParams] = useSearchParams();
    const queryId = searchParams.get('id');
    const navigate = useNavigate();

    const [category, setCategory] = useState('all');
    const [searchQuery, setSearchQuery] = useState(queryId);
    const [loading, setLoading] = useState(true);

    // State to track polling
    const [isPolling, setIsPolling] = useState(true);
    const [pollingCount, setPollingCount] = useState(0);
    const [pollingTimeout, setPollingTimeout] = useState(null);

    // RESPONSES
    // raw
    const [rawResponse, setRawResponse] = useState(null);
    // generated answer
    const [answer, setAnswer] = useState(null);
    // faiss results
    const [results, setResults] = useState([]);

    // Effect to update answer when rawResponse changes
    useEffect(() => {
        if (rawResponse?.answer != null) {
            setAnswer(rawResponse.answer);
            // Stop polling when we have an answer
            setIsPolling(false);
        }
    }, [rawResponse]);

    // Effect to start polling when queryId changes
    useEffect(() => {
        setSearchQuery(queryId);
        setLoading(true);
        setIsPolling(true);
        setAnswer(null);
        setResults([]);
        setRawResponse(null);

        // Initial fetch
        fetchResults(queryId);

        return () => {
            // Clear any existing polling timeout when component unmounts or queryId changes
            if (pollingTimeout) {
                clearTimeout(pollingTimeout);
            }
        };
    }, [queryId]);

    // Fetch results function
    const fetchResults = useCallback(async (id) => {
        if (!id) {
            setLoading(false);
            return;
        }

        try {
            const path = `http://localhost:8080/result?q=${encodeURIComponent(id)}`;
            const response = await FetchData(path);
            console.log("Polling response:", response);

            // Update results if available
            setRawResponse(response);

            if (response.results) {
                setResults(response.results);
                setLoading(false);
            }

            // Continue polling if we don't have an answer yet and polling is enabled
            if (isPolling && !response.answer) {
                const timeout = setTimeout(() => {
                    setPollingCount(prev => prev + 1);
                    fetchResults(id);
                }, POLLING_INTERVAL);

                setPollingTimeout(timeout);
            } else {
                // Stop polling when we have an answer
                setIsPolling(false);
                setLoading(false);
            }
        } catch (error) {
            console.error('Error fetching results:', error);

            // Handle error case but continue polling if needed
            if (isPolling && pollingCount < 30) { // Limit to 30 attempts (1 minute)
                const timeout = setTimeout(() => {
                    setPollingCount(prev => prev + 1);
                    fetchResults(id);
                }, POLLING_INTERVAL);

                setPollingTimeout(timeout);
            } else {
                setIsPolling(false);
                setLoading(false);
            }
        }
    }, [isPolling, pollingCount]);

    const navHome = (event) => {
        event.preventDefault();
        navigate(`/`);
    };

    const handleSearch = (event) => {
        event.preventDefault();
        console.log('Searching for:', searchQuery);
        // Zeroing previous response
        setRawResponse(null);
        setAnswer(null);
        setResults([]);
        navigate(`/search?q=${encodeURIComponent(searchQuery || "")}`);
    };

    return (
        <div className='w-dvh min-h-dvh justify-items-center'>
            <header className='w-full h-20'>
                {/* Header control row */}
                <div className='h-full grid grid-cols-3 place-items-center bg-[#2c243c]'>
                    <div/>
                    <Typography
                        component="button"
                        sx={{color: "white"}}
                        className="justify-self-center"
                        variant="h5"
                        gutterBottom
                        onClick={navHome}
                    >
                        ArXiv RAG Search
                    </Typography>

                    {/* Right header - Info*/}
                    <div className='flex justify-self-start m-5'>
                        <FormatAlignJustifyIcon sx={{color: "white"}} />
                        <Typography sx={{color: "white"}}> Advanced </Typography>

                        <ShuffleIcon className="ml-5" sx={{color: "white"}} />
                        <Typography sx={{color: "white"}}> Random </Typography>

                        <HelpOutlineIcon className="ml-5" sx={{color: "white"}} />
                        <Typography sx={{color: "white"}}> Syntax </Typography>
                    </div>
                </div>
            </header>

            {/* Page Content */}

            {/* Generated RAG LLM answer */}
            <Box className="pt-5 w-[50%] h-full justify-items-center">
                {answer !== null ? (
                    <TypewriterText text={answer} typingSpeed={5} className="min-w-fit"/>
                ) : (
                    <Box className="flex flex-col items-center justify-center w-full pb-56">
                        <CircularProgress color="success" />
                        <LoadingText list={results && Object.keys(results).length > 0 ? RAG_LOADING_MSG : LOADING_MSG} />
                    </Box>
                )}
            </Box>

            <Box sx={{ display: loading ? "none": "block" }}>
                {/* ArXiv Results */}
                <Box
                    className="grid-flow-col grid-cols-${results.length} justify-items-center pt-5 max-w-5xl"
                    sx={{ display: results?.text == null ? "none": "block" }}
                >
                    <List sx={{ listStyle: "decimal", pl: 4 }}>
                        {results?.text?.length > 0 ? (
                            results.text.map((paper, index) => (
                                <ResultArxivItem
                                    key={index}
                                    index={index}
                                    result={paper}
                                />
                            ))
                        ) : (
                            <Typography align="center" sx={{ mt: 4 }}>
                                No results found. Try changing your search terms.
                            </Typography>
                        )}
                    </List>
                </Box>

                {/* Images and Audio */}
                <Card
                    sx={{display: (results?.audio == null && results?.image == null) ? "none": "block"}}
                    className="max-w-5xl w-full min-w-full justify-items-center pt-5"
                >
                    <Typography variant='h3' className='pt-2'>
                        {"Images and audio content"}
                    </Typography>

                    <Divider className="pt-2" sx={{width: "95%", height:"2px", color: "#2c243c"}}/>

                    <Card className="w-full pb-5 pt-2 pl-1 justify-items-center">
                        <ImageDisplay images={results?.image} className="pb-5"/>
                        <Divider className="pt-2" sx={{width: "95%", height:"2px", color: "#2c243c"}}/>
                        <AudioDisplay audioFiles={results?.audio}/>
                    </Card>
                </Card>

                {/* Youtube Videos */}
                <Box
                    sx={{display: results?.video == null ? "none": "block"}}
                    className="grid-flow-col max-w-5xl min-w-full justify-items-center pt-5 mt-7"
                >
                    <Typography className='w-fit justify-self-start' variant="h4">
                        {"You may find interest in content of"}
                        <br/>
                        <i className="pl-10">{"Popular Scientific Content Creators"}</i>
                    </Typography>

                    <Divider className="pt-2" sx={{width: "95%", height:"2px", color: "#2c243c"}}/>

                    <List className="flex" sx={{ listStyle: "decimal", pl: 4 }}>
                        {results?.video?.length > 0 ? (
                            results.video.slice(0, 3).map((video, index) => (
                                <ResultItemVideo key={index} video={video}/>
                            ))
                        ): ""}
                    </List>
                </Box>
            </Box>
        </div>
    );
}

export default ResultPage;