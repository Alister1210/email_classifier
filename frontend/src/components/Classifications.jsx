import React, { useState, useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import api from "../services/api";
import { Button } from "./ui/button";
import { Alert } from "./ui/alert";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Spinner } from "./ui/spinner";
import { Table as SimpleTable, THead, TBody, TR, TH, TD } from "./ui/table";
import { motion, AnimatePresence } from "framer-motion";
import { useGmailConnection } from "../hooks/useGmailConnection";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  flexRender,
} from "@tanstack/react-table";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import { RefreshCw, Search, TrendingUp, Mail } from "lucide-react";

const LABEL_COLORS = {
  finance: "#3b82f6", // blue
  personal: "#10b981", // emerald
  promotions: "#f59e0b", // amber
  spam: "#ef4444", // red
  "travel updates": "#8b5cf6", // purple
};

const LABEL_DISPLAY_NAMES = {
  finance: "Finance",
  personal: "Personal",
  promotions: "Promotions",
  spam: "Spam",
  "travel updates": "Travel Updates",
};

const Classifications = () => {
  const [classifications, setClassifications] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [warning, setWarning] = useState(null);
  const [globalFilter, setGlobalFilter] = useState("");
  const [refreshing, setRefreshing] = useState(false);

  const navigate = useNavigate();
  const { isConnected } = useGmailConnection();

  // Redirect if not connected
  useEffect(() => {
    if (!isConnected) {
      navigate("/");
    }
  }, [isConnected, navigate]);

  const fetchClassifications = async (isRefresh = false) => {
    if (isRefresh) {
      setRefreshing(true);
    } else {
      setLoading(true);
    }
    setError(null);
    setWarning(null);

    try {
      const response = await api.get("/api/results", { timeout: 8000 });

      // Check for duplicates
      const messageIds = response.data.classifications.map(
        (item) => item.message_id
      );
      const duplicates = messageIds.filter(
        (id, index) => messageIds.indexOf(id) !== index
      );

      if (duplicates.length > 0) {
        setWarning(
          `Found ${duplicates.length} duplicate entries (showing latest)`
        );
      }

      // Remove duplicates, keep latest by timestamp
      const uniqueClassifications = Object.values(
        response.data.classifications.reduce((acc, item) => {
          const existing = acc[item.message_id];
          if (
            !existing ||
            new Date(item.timestamp) > new Date(existing.timestamp)
          ) {
            acc[item.message_id] = item;
          }
          return acc;
        }, {})
      ).sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

      setClassifications(uniqueClassifications);
    } catch (err) {
      console.error("Fetch classifications error:", err);
      if (err.code === "ERR_NETWORK") {
        setError(
          "Cannot connect to backend server. Please ensure it's running."
        );
      } else {
        setError(`Failed to fetch classifications: ${err.message}`);
      }
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchClassifications();
    const interval = setInterval(() => fetchClassifications(true), 30000);
    return () => clearInterval(interval);
  }, []);

  const columns = useMemo(
    () => [
      {
        accessorKey: "sender",
        header: "Sender",
        cell: (info) => (
          <div className="max-w-xs truncate font-medium">{info.getValue()}</div>
        ),
      },
      {
        accessorKey: "subject",
        header: "Subject",
        cell: (info) => (
          <div className="max-w-sm truncate">{info.getValue()}</div>
        ),
      },
      {
        accessorKey: "label",
        header: "Classification",
        cell: (info) => {
          const label = info.getValue()?.toLowerCase();
          const displayName = LABEL_DISPLAY_NAMES[label] || label;
          const color = LABEL_COLORS[label] || "#6b7280";

          return (
            <span
              className="inline-flex px-3 py-1 rounded-full text-xs font-medium text-white"
              style={{ backgroundColor: color }}
            >
              {displayName}
            </span>
          );
        },
      },
      {
        accessorKey: "confidence",
        header: "Confidence",
        cell: (info) => {
          const confidence = info.getValue();
          const percentage = Math.round(confidence * 100);
          const getColor = () => {
            if (percentage >= 90) return "text-emerald-600";
            if (percentage >= 70) return "text-yellow-600";
            return "text-red-600";
          };
          return (
            <span className={`font-medium ${getColor()}`}>{percentage}%</span>
          );
        },
      },
      {
        accessorKey: "timestamp",
        header: "Time",
        cell: (info) => {
          const date = new Date(info.getValue());
          return (
            <div className="text-sm">
              <div>{date.toLocaleDateString()}</div>
              <div className="text-gray-500">{date.toLocaleTimeString()}</div>
            </div>
          );
        },
      },
    ],
    []
  );

  const table = useReactTable({
    data: classifications,
    columns,
    state: { globalFilter },
    onGlobalFilterChange: setGlobalFilter,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    initialState: {
      pagination: {
        pageSize: 10,
      },
    },
  });

  const stats = useMemo(() => {
    const total = classifications.length;
    const labelCounts = classifications.reduce((acc, c) => {
      const label = c.label?.toLowerCase();
      acc[label] = (acc[label] || 0) + 1;
      return acc;
    }, {});

    const spam = labelCounts.spam || 0;
    const notSpam = total - spam;
    const avgConfidence = total
      ? classifications.reduce((acc, c) => acc + (c.confidence || 0), 0) / total
      : 0;

    return { total, spam, notSpam, labelCounts, avgConfidence };
  }, [classifications]);

  const chartData = useMemo(() => {
    const last7Days = Array.from({ length: 7 }, (_, i) => {
      const date = new Date();
      date.setDate(date.getDate() - (6 - i));
      const dayData = {
        day: date.toLocaleDateString("en-US", { weekday: "short" }),
        date: date.toDateString(),
        spam: 0,
        notSpam: 0,
      };

      // Initialize all label counts
      Object.keys(LABEL_COLORS).forEach((label) => {
        dayData[label] = 0;
      });

      return dayData;
    });

    classifications.forEach((c) => {
      const emailDate = new Date(c.timestamp).toDateString();
      const dayData = last7Days.find((d) => d.date === emailDate);
      if (dayData) {
        const label = c.label?.toLowerCase();
        if (label === "spam") {
          dayData.spam += 1;
        } else {
          dayData.notSpam += 1;
        }

        // Count specific labels
        if (dayData.hasOwnProperty(label)) {
          dayData[label] += 1;
        }
      }
    });

    return last7Days;
  }, [classifications]);

  // Pie chart data for all labels
  const labelPieData = Object.entries(stats.labelCounts).map(
    ([label, count]) => ({
      name: LABEL_DISPLAY_NAMES[label] || label,
      value: count,
      color: LABEL_COLORS[label] || "#6b7280",
    })
  );

  // Spam vs Non-spam pie data
  const spamPieData = [
    { name: "Spam", value: stats.spam, color: "#ef4444" },
    { name: "Not Spam", value: stats.notSpam, color: "#10b981" },
  ];

  if (loading && !refreshing) {
    return (
      <div className="flex items-center justify-center py-16">
        <div className="text-center space-y-4">
          <Spinner className="w-8 h-8" />
          <p className="text-gray-600 dark:text-gray-400">
            Loading classifications...
          </p>
        </div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
          Email Classifications
        </h1>
        <Button
          onClick={() => fetchClassifications(true)}
          disabled={refreshing}
          variant="outline"
          className="flex items-center gap-2"
        >
          <RefreshCw size={16} className={refreshing ? "animate-spin" : ""} />
          {refreshing ? "Refreshing..." : "Refresh"}
        </Button>
      </div>

      {/* Alerts */}
      {error && <Alert variant="destructive">{error}</Alert>}
      {warning && <Alert variant="default">{warning}</Alert>}

      {classifications.length === 0 ? (
        <Card>
          <CardContent className="py-16 text-center">
            <Mail size={48} className="mx-auto text-gray-400 mb-4" />
            <h3 className="text-lg font-semibold mb-2">
              No Classifications Yet
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-6">
              Connect your Gmail account and wait for emails to be processed.
            </p>
            <Button onClick={() => navigate("/")} variant="outline">
              Go to Homepage
            </Button>
          </CardContent>
        </Card>
      ) : (
        <>
          {/* Stats Cards */}
          <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
            <StatsCard
              title="Total Emails"
              value={stats.total}
              icon={<Mail size={20} />}
              color="bg-blue-500"
            />
            <StatsCard
              title="Spam"
              value={stats.labelCounts.spam || 0}
              icon={<TrendingUp size={20} />}
              color="bg-red-500"
            />
            <StatsCard
              title="Finance"
              value={stats.labelCounts.finance || 0}
              icon={<TrendingUp size={20} />}
              color="bg-blue-500"
            />
            <StatsCard
              title="Personal"
              value={stats.labelCounts.personal || 0}
              icon={<TrendingUp size={20} />}
              color="bg-emerald-500"
            />
            <StatsCard
              title="Promotions"
              value={stats.labelCounts.promotions || 0}
              icon={<TrendingUp size={20} />}
              color="bg-amber-500"
            />
            <StatsCard
              title="Travel"
              value={stats.labelCounts["travel updates"] || 0}
              icon={<TrendingUp size={20} />}
              color="bg-purple-500"
            />
          </div>

          {/* Charts */}
          <div className="grid md:grid-cols-3 gap-6">
            {/* Daily Activity Chart */}
            <Card className="md:col-span-2">
              <CardHeader>
                <CardTitle className="text-lg">
                  Daily Activity (Last 7 Days)
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData}>
                      <XAxis
                        dataKey="day"
                        fontSize={12}
                        tickLine={false}
                        axisLine={false}
                      />
                      <YAxis
                        fontSize={12}
                        tickLine={false}
                        axisLine={false}
                        allowDecimals={false}
                      />
                      <Tooltip />
                      <Legend />
                      <Bar
                        dataKey="finance"
                        stackId="a"
                        fill={LABEL_COLORS.finance}
                        name="Finance"
                      />
                      <Bar
                        dataKey="personal"
                        stackId="a"
                        fill={LABEL_COLORS.personal}
                        name="Personal"
                      />
                      <Bar
                        dataKey="promotions"
                        stackId="a"
                        fill={LABEL_COLORS.promotions}
                        name="Promotions"
                      />
                      <Bar
                        dataKey="spam"
                        stackId="a"
                        fill={LABEL_COLORS.spam}
                        name="Spam"
                      />
                      <Bar
                        dataKey="travel updates"
                        stackId="a"
                        fill={LABEL_COLORS["travel updates"]}
                        name="Travel Updates"
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Spam vs Non-Spam Distribution */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Spam Detection</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={spamPieData}
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        dataKey="value"
                        label={({ name, percent }) =>
                          `${name} ${(percent * 100).toFixed(0)}%`
                        }
                      >
                        {spamPieData.map((entry, index) => (
                          <Cell key={index} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* All Labels Distribution */}
            <Card className="md:col-span-3">
              <CardHeader>
                <CardTitle className="text-lg">
                  Classification Distribution
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={labelPieData}
                        cx="50%"
                        cy="50%"
                        outerRadius={100}
                        dataKey="value"
                        label={({ name, percent }) =>
                          `${name} ${(percent * 100).toFixed(0)}%`
                        }
                      >
                        {labelPieData.map((entry, index) => (
                          <Cell key={index} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Data Table */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">
                  Recent Classifications
                </CardTitle>
                <div className="flex items-center gap-4">
                  <div className="relative">
                    <Search
                      size={16}
                      className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400"
                    />
                    <input
                      className="pl-10 h-9 w-64 rounded-md border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 px-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                      placeholder="Search emails..."
                      value={globalFilter ?? ""}
                      onChange={(e) => setGlobalFilter(e.target.value)}
                    />
                  </div>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="rounded-md border">
                  <SimpleTable>
                    <THead>
                      {table.getHeaderGroups().map((headerGroup) => (
                        <TR key={headerGroup.id}>
                          {headerGroup.headers.map((header) => (
                            <TH
                              key={header.id}
                              onClick={header.column.getToggleSortingHandler()}
                              className="cursor-pointer select-none hover:bg-gray-50 dark:hover:bg-gray-800"
                            >
                              <div className="flex items-center gap-1">
                                {flexRender(
                                  header.column.columnDef.header,
                                  header.getContext()
                                )}
                                {header.column.getIsSorted() === "asc" && "↑"}
                                {header.column.getIsSorted() === "desc" && "↓"}
                              </div>
                            </TH>
                          ))}
                        </TR>
                      ))}
                    </THead>
                    <TBody>
                      <AnimatePresence initial={false}>
                        {table.getRowModel().rows.map((row, index) => (
                          <motion.tr
                            key={row.id}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            transition={{ delay: index * 0.05 }}
                            className="hover:bg-gray-50 dark:hover:bg-gray-800/50"
                          >
                            {row.getVisibleCells().map((cell) => (
                              <TD key={cell.id}>
                                {flexRender(
                                  cell.column.columnDef.cell,
                                  cell.getContext()
                                )}
                              </TD>
                            ))}
                          </motion.tr>
                        ))}
                      </AnimatePresence>
                    </TBody>
                  </SimpleTable>
                </div>

                {/* Pagination */}
                <div className="flex items-center justify-between">
                  <div className="text-sm text-gray-600 dark:text-gray-400">
                    Showing {table.getRowModel().rows.length} of{" "}
                    {classifications.length} results
                  </div>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => table.setPageIndex(0)}
                      disabled={!table.getCanPreviousPage()}
                    >
                      First
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => table.previousPage()}
                      disabled={!table.getCanPreviousPage()}
                    >
                      Previous
                    </Button>
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      Page {table.getState().pagination.pageIndex + 1} of{" "}
                      {table.getPageCount()}
                    </span>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => table.nextPage()}
                      disabled={!table.getCanNextPage()}
                    >
                      Next
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() =>
                        table.setPageIndex(table.getPageCount() - 1)
                      }
                      disabled={!table.getCanNextPage()}
                    >
                      Last
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </motion.div>
  );
};

function StatsCard({ title, value, icon, color }) {
  return (
    <Card className="overflow-hidden">
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400 font-medium">
              {title}
            </p>
            <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              {value}
            </p>
          </div>
          <div className={`p-3 rounded-full ${color} text-white`}>{icon}</div>
        </div>
      </CardContent>
    </Card>
  );
}

export default Classifications;
